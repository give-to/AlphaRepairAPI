import json
import os
import re
import subprocess
import time
import torch
import argparse

from transformers import RobertaTokenizer, RobertaForMaskedLM

from simple_template import generate_template, remove_redudant, generate_match_template, match_simple_operator
from tool.logger import Logger
from tool.fault_localization import get_location
from tool.d4j import build_d4j1_2
from validate_patches import GVpatches, UNIAPRpatches
from bert_beam_search import BeamSearch

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def comment_remover(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


def add_new_line(logger, file, line_loc, tokenizer, model, beam_width, re_rank=True, top_n_patches=-1):
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        data = f.readlines()

    ret_before = []
    mask_token = "<mask>"
    pre_code = data[:line_loc]
    post_code = data[line_loc:]
    post_code1 = data[line_loc + 1:]
    old_code = data[line_loc].strip()
    masked_line = " " + mask_token * 20 + " "
    line_size = 100
    while (1):
        pre_code_input = "</s> " + " ".join(
            [x.strip() for x in pre_code[-line_size:]])
        post_code_input = " ".join([x.strip() for x in post_code[0:line_size]]).replace("\n", "").strip()
        if tokenizer(pre_code_input + masked_line + post_code_input, return_tensors='pt')['input_ids'].size()[1] < 490:
            break
        line_size -= 1

    print(">>>>> Begin Some Very Long Beam Generation <<<<<")
    print("Context Line Size: {}".format(line_size))  # actual context len =  2*line_size
    print("Context Before:\n{}".format(pre_code_input))
    print("Context After:\n{}".format(post_code_input))

    generate_time_start = time.time()
    # Straight up line replacement
    for token_len in range(1, 30):  # Within 10
        if judge_patches_num(ret_before):
            break
        masked_line = " " + mask_token * token_len + " "
        beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                 beam_width=beam_width, re_rank=re_rank)
        beam_list, masked_index = beam_engine.generate_beam()
        for beam in beam_list:
            if judge_patches_num(ret_before):
                break
            ret_before.append(("".join(beam[2]), beam[0] / token_len, "Before " + masked_line))
    logger.logo("Patch \"add_new_line part\" Generation Time:" + str(time.time() - generate_time_start))
    generate_time_start = time.time()

    ret_before.sort(key=lambda x: x[1], reverse=True)
    ret_before = remove_redudant(ret_before)

    ret = []
    ret.extend(ret_before)
    ret.sort(key=lambda x: x[1], reverse=True)

    if top_n_patches == -1:
        return pre_code, old_code, ret, post_code1
    else:
        return pre_code, old_code, ret[:top_n_patches], post_code1


def judge_patches_num(ret, max_patch=5000):
    if (len(ret) > max_patch):
        return True
    return False


def process_file(logger, file, line_loc, tokenizer, model, beam_width, re_rank=True, top_n_patches=-1):
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        data = f.readlines()

    ret = []
    mask_token = "<mask>"
    pre_code = data[:line_loc]
    fault_line = comment_remover(data[line_loc].strip())  # remove comments
    old_code = data[line_loc].strip()
    post_code = data[line_loc + 1:]

    line_size = 100
    while (1):
        pre_code_input = "</s> " + " ".join([x.strip() for x in pre_code[-line_size:]])
        post_code_input = " ".join([x.strip() for x in post_code[0:line_size]]).replace("\n", "").strip()
        if tokenizer(pre_code_input + fault_line + post_code_input, return_tensors='pt')['input_ids'].size()[1] < 490:
            break
        line_size -= 1

    print(">>>>> Begin Some Very Long Beam Generation <<<<<")
    print("Context Line Size: {}".format(line_size))  # actual context len =  2*line_size
    print("Context Before:\n{}".format(pre_code_input))
    print(">> {} <<".format(fault_line))
    print("Context After:\n{}".format(post_code_input))

    fault_line_token_size = tokenizer(fault_line, return_tensors='pt')["input_ids"].shape[1] - 2

    # Straight up line replacement
    generate_time_start = time.time()

    for token_len in range(fault_line_token_size - 5, fault_line_token_size + 5):  # Within 10
        if judge_patches_num(ret):
            break
        if token_len <= 0:
            continue
        masked_line = " " + mask_token * token_len + " "
        beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                 beam_width=beam_width, re_rank=re_rank)
        beam_list, masked_index = beam_engine.generate_beam()
        for beam in beam_list:
            if judge_patches_num(ret):
                break
            ret.append(("".join(beam[2]), beam[0] / token_len, masked_line))

    logger.logo("Patch \"token_len part\" Generation Time:" + str(time.time() - generate_time_start))
    generate_time_start = time.time()

    templates = generate_template(fault_line)
    for partial_beginning, partial_end in templates:
        temp_size = fault_line_token_size - (
                tokenizer(partial_beginning, return_tensors='pt')["input_ids"].shape[1] - 2) - (
                            tokenizer(partial_end, return_tensors='pt')["input_ids"].shape[1] - 2)
        for token_len in range(2, 11):
            if judge_patches_num(ret):
                break
            if token_len <= 0:
                continue
            masked_line = " " + partial_beginning + mask_token * token_len + partial_end + " "
            beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                     beam_width=beam_width, re_rank=re_rank)
            beam_list, masked_index = beam_engine.generate_beam()
            for beam in beam_list:
                if judge_patches_num(ret):
                    break
                ret.append((partial_beginning + "".join(beam[2]) + partial_end, beam[0] / token_len, masked_line))

    logger.logo("Patch \"templates part\" Generation Time:" + str(time.time() - generate_time_start))
    generate_time_start = time.time()

    match_template = generate_match_template(fault_line, tokenizer)
    for match, length in match_template:
        if judge_patches_num(ret):
            break
        for token_len in range(1, length + 5):
            if judge_patches_num(ret):
                break
            if len(match.split(mask_token)) == 2:
                masked_line = " " + match.split(mask_token)[0] + mask_token * token_len + match.split(mask_token)[
                    1] + " "
                beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                         beam_width=beam_width, re_rank=re_rank)
                beam_list, masked_index = beam_engine.generate_beam()
                for beam in beam_list:
                    if judge_patches_num(ret):
                        break
                    ret.append((match.split(mask_token)[0] + "".join(beam[2]) + match.split(mask_token)[1],
                                beam[0] / token_len, masked_line))
            else:
                if judge_patches_num(ret):
                    break
                masked_line = " "
                masked_line += (mask_token * token_len).join(match.split(mask_token)) + " "
                beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                         beam_width=beam_width, re_rank=re_rank)
                beam_list, masked_index = beam_engine.generate_beam()
                for beam in beam_list:
                    if judge_patches_num(ret):
                        break
                    index = 0
                    gen_line = ""
                    for c in masked_line.split(mask_token)[:-1]:
                        gen_line += c
                        gen_line += beam[2][index]
                        index += 1
                    gen_line += masked_line.split(mask_token)[-1]
                    gen_line = gen_line[1:-1]
                    ret.append((gen_line, beam[0] / (token_len * (len(match.split(mask_token)) - 1)), masked_line))

    logger.logo("Patch \"match_template part\" Generation Time:" + str(time.time() - generate_time_start))
    generate_time_start = time.time()

    simple_operator_template = match_simple_operator(fault_line, tokenizer)
    for template in simple_operator_template:
        if judge_patches_num(ret):
            break
        token_len = template.count("<mask>")
        masked_line = " " + template + " "
        beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                 beam_width=beam_width, re_rank=re_rank)
        beam_list, masked_index = beam_engine.generate_beam()
        for beam in beam_list:
            if judge_patches_num(ret):
                break
            index = 0
            gen_line = ""
            for c in masked_line.split(mask_token)[:-1]:
                gen_line += c
                gen_line += beam[2][index]
                index += 1
            gen_line += masked_line.split(mask_token)[-1]
            gen_line = gen_line[1:-1]
            ret.append((gen_line, beam[0] / token_len, masked_line))

    logger.logo("Patch \"simple_operator_template part\" Generation Time:" + str(time.time() - generate_time_start))
    generate_time_start = time.time()

    ret.sort(key=lambda x: x[1], reverse=True)
    ret = remove_redudant(ret)
    if top_n_patches == -1:
        return pre_code, old_code, ret, post_code
    else:
        return pre_code, old_code, ret[:top_n_patches], post_code


def write_changes(dst_root, changes):
    # write_list("tmpStore/{}/{}/{}/gen_line".format(bug_id, file.replace("/", "."), line_number), changes1)
    # write_list("tmpStore/{}/{}/{}/prob".format(bug_id, file.replace("/", "."), line_number), changes1)
    # write_list("tmpStore/{}/{}/{}/masked_line".format(bug_id, file.replace("/", "."), line_number), changes1)
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    f_change = open(dst_root + "/change", "w")
    f_prob = open(dst_root + "/prob", 'w')
    f_maskedline = open(dst_root + "/maskedline", 'w')
    for change, prob, masked_line in changes:
        if change.endswith("\n"):
            f_change.write(change)
        else:
            f_change.write(change+'\n')

        f_prob.write(str(prob)+'\n')

        if masked_line.endswith("\n"):
            f_maskedline.write(masked_line)
        else:
            f_maskedline.write(masked_line+'\n')
    f_change.close()
    f_prob.close()
    f_maskedline.close()


def read_changes(store_root):
    with open(store_root + "/change", 'r', encoding='utf-8', errors='ignore') as f1:
        data_change = f1.readlines()
    with open(store_root + "/prob", 'r', encoding='utf-8', errors='ignore') as f2:
        data_prob = f2.readlines()
    with open(store_root + "/maskedline", 'r', encoding='utf-8', errors='ignore') as f3:
        data_maskedline = f3.readlines()
    ret = []
    for i in range(0, len(data_change)):
        ret.append((data_change[i].strip(), data_prob[i].strip(), data_maskedline[i].strip()))
    return ret


def main(validation, bug_ids, output_folder, skip_validation, uniapr, beam_width, re_rank, perfect, top_n_patches):
    if bug_ids[0] == 'none':
        bug_ids = build_d4j1_2()

    model = RobertaForMaskedLM.from_pretrained("codebert-base-mlm").to(device)
    tokenizer = RobertaTokenizer.from_pretrained("codebert-base-mlm")

    for bug_id in bug_ids:
        subprocess.run('rm -rf ' + '/tmp/' + bug_id, shell=True)
        # subprocess.run("defects4j checkout -p %s -v %s -w %s" % (
        #     bug_id.split('-')[0], bug_id.split('-')[1] + 'b', ('/tmp/' + bug_id)), shell=True)
        subprocess.run("catena4j checkout -p %s -v %s -w %s" % (
            bug_id.split('-')[0], bug_id.split('-')[1] + 'b' + bug_id.split('-')[2], ('/tmp/' + bug_id)), shell=True)
        patch_pool_folder = "patches-pool"
        location = get_location(bug_id, perfect=perfect)
        # location = get_location_tbar(bug_id)
        if perfect:
            patch_pool_folder = "pfl-patches-pool-temp"

        testmethods = os.popen('defects4j test -w %s' % ('/tmp/' + bug_id)).readlines()
        testmethods.pop(0)
        for i in range(0,len(testmethods)):
            testmethods[i] = testmethods[i].replace("  - ", "")
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        logger = Logger(output_folder + '/' + bug_id + "_result.txt")
        logger.logo(args)
        if uniapr:
            validator = UNIAPRpatches(bug_id, testmethods, logger, patch_pool_folder=patch_pool_folder,
                                      skip_validation=skip_validation)
        else:
            validator = GVpatches(bug_id, testmethods, logger, patch_pool_folder=patch_pool_folder,
                                  skip_validation=skip_validation)

        for file, line_number in location:
            print('Location: {} line # {}'.format(file, line_number))
            file = '/tmp/' + bug_id + '/' + file

            start_time = time.time()
            print(validation)
            if not validation:
                if len(location) <= 3 and perfect:  # too many lines, can't really handle in time
                    pre_code, fault_line, changes1, post_code = add_new_line(logger, file, line_number, tokenizer,
                                                                             model, 25,  # default 15; paper is 25
                                                                             re_rank, top_n_patches)
                    pre_code, fault_line, changes, post_code = process_file(logger, file, line_number, tokenizer, model,
                                                                            25,
                                                                            # default 15; paper is 25
                                                                            re_rank, top_n_patches)
                else:
                    pre_code, fault_line, changes1, post_code = add_new_line(logger, file, line_number, tokenizer,
                                                                             model, beam_width,
                                                                             re_rank, top_n_patches)
                    pre_code, fault_line, changes, post_code = process_file(logger, file, line_number, tokenizer, model,
                                                                            beam_width,
                                                                            # default 15; paper is 25
                                                                            re_rank, top_n_patches)
                end_time = time.time()
                generate_time = end_time - start_time
                changes1.extend(changes)
                changes1.sort(key=lambda x: x[1], reverse=True)
                logger.logo("Patches Generated at {}:{} Total Number:{}".format(file, line_number + 1, len(changes1)))
                logger.logo(
                    "Patches Generated at {}:{} Total Time:{}".format(file, line_number + 1, end_time - start_time))
                # save info to file
                dst_root = "store_changes/{}/{}/{}".format(bug_id, file.replace("/", "."), line_number)
                write_changes(dst_root, changes1)
                f_time = open(dst_root + "/time", 'w')
                f_time.write(str(end_time - start_time))
                f_time.close()

            else:
                changes1 = read_changes("store_changes/{}/{}/{}".format(bug_id, file.replace("/", "."), line_number))
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    data = f.readlines()
                pre_code = data[:line_number]
                fault_line = data[line_number].strip()
                post_code = data[line_number + 1:]
                dst_root = "store_changes/{}/{}/{}".format(bug_id, file.replace("/", "."), line_number)
                with open(dst_root + "/time", 'r', encoding='utf-8', errors='ignore') as f:
                    generate_time = f.readlines()[0]

            validator.add_new_patch_generation(pre_code, fault_line, changes1, post_code, file, line_number,
                                               float(generate_time))
        if validation:
            validator.validate()

        # subprocess.run('rm -rf ' + '/tmp/' + bug_id, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation', action='store_true', default=False)
    parser.add_argument('--bug_id', type=str, default='Closure-2')
    parser.add_argument('--uniapr', action='store_true', default=False)
    parser.add_argument('--output_folder', type=str, default='codebert_result')
    parser.add_argument('--skip_v', action='store_true', default=False)
    parser.add_argument('--re_rank', action='store_true', default=False)
    parser.add_argument('--beam_width', type=int, default=5)  # perfect: 15; paper is 25
    parser.add_argument('--perfect', action='store_true', default=False)
    parser.add_argument('--top_n_patches', type=int, default=-1)
    args = parser.parse_args()
    print("Run with setting:")
    print(args)
    main(args.validation, [args.bug_id], args.output_folder, args.skip_v, args.uniapr, args.beam_width,
         args.re_rank, args.perfect, args.top_n_patches)
