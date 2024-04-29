import argparse
import os
import signal
import subprocess
import time
from tool.logger import Logger

def write_time(bug_id, time):
    if not os.path.exists("time_info"):
        os.mkdir("time_info")
    f_time = open("time_info/" + bug_id, 'w')
    f_time.write(str(time))
    f_time.close()
def read_time(bug_id):
    with open("time_info/" + bug_id, 'r', encoding='utf-8', errors='ignore') as f:
        generate_time = f.readlines()
    return generate_time[0]
def main(start_bug_index, end_bug_index, validation):
    buglist = open("buglist", 'r').readlines()
    if not validation:
        logger = Logger(str(start_bug_index) + "-" + str(end_bug_index) + "_timelog.txt")
        for i in range(start_bug_index, end_bug_index + 1):
            bugid = buglist[i].strip()
            print("---------------- Now is Generating " + bugid + " ----------------")
            cmd = 'python3 experiment.py --bug_id=%s' % (bugid)
            Returncode = ""

            begin_time = time.time()
            local_time = time.localtime(begin_time)
            logger.logo("----------------------------------------")
            logger.logo("BugId: " + bugid)
            logger.logo("Generating Start Time: " + "{}:{}:{}".format(str(local_time.tm_hour), str(local_time.tm_min), str(local_time.tm_sec)))
            error_file = open(str(start_bug_index) + "-" + str(end_bug_index) +"_stderr.txt", "ab")
            child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=error_file, bufsize=-1,
                                     start_new_session=True)

            while True:
                Flag = child.poll()
                if Flag == 0:
                    line = child.stdout.readline().decode("utf8")
                    print(line)
                    error_file.close()
                    logger.logo("Exit With " + str(Flag) + "!")
                    end_time = time.time()
                    local_time = time.localtime(end_time)
                    logger.logo("Generating End Time: " + "{}:{}:{}".format(str(local_time.tm_hour), str(local_time.tm_min),
                                                                 str(local_time.tm_sec)))
                    total_time = time.time() - begin_time
                    write_time(bugid, total_time)
                    logger.logo("Generating Total Time: {}min".format(str(round(total_time / 60))))
                    break
                elif Flag != 0 and Flag is not None:
                    line = child.stdout.readline().decode("utf8")
                    print(line)
                    error_file.close()
                    logger.logo("Exit With " + str(Flag) + "...")
                    end_time = time.time()
                    local_time = time.localtime(end_time)
                    logger.logo("Generating End Time: " + "{}:{}:{}".format(str(local_time.tm_hour), str(local_time.tm_min),
                                                                 str(local_time.tm_sec)))
                    total_time = time.time() - begin_time
                    write_time(bugid, total_time)
                    logger.logo("Generating Total Time: {}min".format(str(round(total_time / 60))))
                    break
                elif time.time() - begin_time > 18000:
                    line = child.stdout.readline().decode("utf8")
                    print(line)
                    error_file.close()
                    logger.logo("Time Out!!!")
                    end_time = time.time()
                    local_time = time.localtime(end_time)
                    os.killpg(os.getpgid(child.pid), signal.SIGTERM)
                    logger.logo("Generating End Time: " + "{}:{}:{}".format(str(local_time.tm_hour), str(local_time.tm_min),
                                                                 str(local_time.tm_sec)))
                    total_time = time.time() - begin_time
                    write_time(bugid, total_time)
                    logger.logo("Generating Total Time: {}min".format(str(round(total_time/60))))
                    break
                else:
                    time.sleep(0.001)
                    line = child.stdout.readline().decode("utf8")
                    print(line)
        print("---------------- Finish Generating Running ----------------")
    else:
        logger = Logger(str(start_bug_index) + "-" + str(end_bug_index) + "_validate_timelog.txt")
        for i in range(start_bug_index, end_bug_index + 1):
            bugid = buglist[i].strip()
            print("---------------- Now is Validating " + bugid + " ----------------")
            cmd = 'python3 experiment.py --bug_id=%s --validation' % (bugid)
            Returncode = ""

            begin_time = time.time()
            local_time = time.localtime(begin_time)
            logger.logo("----------------------------------------")
            logger.logo("BugId: " + bugid)
            logger.logo("Validating Start Time: " + "{}:{}:{}".format(str(local_time.tm_hour), str(local_time.tm_min),
                                                                      str(local_time.tm_sec)))
            error_file = open(str(start_bug_index) + "-" + str(end_bug_index) + "_stderr.txt", "ab")
            child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=error_file, bufsize=-1,
                                     start_new_session=True)
            generating_time = float(read_time(bugid))
            while True:
                Flag = child.poll()
                if Flag == 0:
                    line = child.stdout.readline().decode("utf8")
                    print(line)
                    error_file.close()
                    logger.logo("Exit With " + str(Flag) + "!")
                    end_time = time.time()
                    local_time = time.localtime(end_time)
                    logger.logo(
                        "Validating End Time: " + "{}:{}:{}".format(str(local_time.tm_hour), str(local_time.tm_min),
                                                                    str(local_time.tm_sec)))
                    total_time = time.time() - begin_time
                    logger.logo("Generating Total Time: {}min".format(str(round(generating_time/60))))
                    logger.logo("Validating Total Time: {}min".format(str(round(total_time / 60))))
                    logger.logo("Total Time: {}min".format(str(round((total_time + generating_time)/ 60))))
                    break
                elif Flag != 0 and Flag is not None:
                    line = child.stdout.readline().decode("utf8")
                    print(line)
                    error_file.close()
                    logger.logo("Exit With " + str(Flag) + "...")
                    end_time = time.time()
                    local_time = time.localtime(end_time)
                    logger.logo(
                        "Validating End Time: " + "{}:{}:{}".format(str(local_time.tm_hour), str(local_time.tm_min),
                                                                    str(local_time.tm_sec)))
                    total_time = time.time() - begin_time
                    logger.logo("Generating Total Time: {}min".format(str(round(generating_time/60))))
                    logger.logo("Validating Total Time: {}min".format(str(round(total_time / 60))))
                    logger.logo("Total Time: {}min".format(str(round((total_time + generating_time)/ 60))))
                    break
                elif time.time() - begin_time + generating_time > 18000:
                    line = child.stdout.readline().decode("utf8")
                    print(line)
                    error_file.close()
                    logger.logo("Time Out!!!")
                    local_time = time.localtime(time.time())
                    os.killpg(os.getpgid(child.pid), signal.SIGTERM)
                    logger.logo(
                        "Validating End Time: " + "{}:{}:{}".format(str(local_time.tm_hour), str(local_time.tm_min),
                                                                    str(local_time.tm_sec)))
                    total_time = time.time() - begin_time
                    logger.logo("Generating Total Time: {}min".format(str(round(generating_time/60))))
                    logger.logo("Validating Total Time: {}min".format(str(round(total_time / 60))))
                    logger.logo("Total Time: {}min".format(str(round((total_time + generating_time)/ 60))))
                    break
                else:
                    time.sleep(0.001)
                    line = child.stdout.readline().decode("utf8")
                    print(line)
        print("---------------- Finish Validating Running ----------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_bug_index', type=int, default=0)
    parser.add_argument('--end_bug_index', type=int, default=0)
    parser.add_argument('--validation', type=bool, default=False)
    args = parser.parse_args()
    print("Run with setting:")
    print(args)
    main(args.start_bug_index, args.end_bug_index, args.validation)
