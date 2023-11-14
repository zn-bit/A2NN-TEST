import numpy as np
import os


if __name__ == '__main__':
    # result_bin_0 = np.fromfile("./vggoutput_pack_U2.bin", dtype=np.int8)
    result_bin_0 = np.fromfile("./data/output_bin/vggoutput_pack.bin", dtype=np.int8)
    result_bin = []
    for i in range(0, result_bin_0.shape[0], 8):
        result_bin.append(result_bin_0[i])
    with open("./data/output_bin/UC_test_0.2.txt") as f:
        f_result_right = f.readlines()
    result_right = []
    imglist = []
    for line in f_result_right:
        img_tmp, result_tmp = line.split()
        result_right.append(int(result_tmp))
        imglist.append(img_tmp)
    cnt_all = len(result_right)
    print("*"*50)
    print("待测图像总数： " + str(cnt_all))
    result_right = np.array(result_right)
    result_bin = np.array(result_bin)
    cnt_right_list = result_right==result_bin
    cnt_right = sum(cnt_right_list)
    print("正确分类图像数： "+str(cnt_right))
    rate = cnt_right/cnt_all
    print("分类准确率： "+str(rate))
    wronglist_bin = []
    wronglist_true = []
    wronglist_img = []
    show_list = []
    with open("wrong.txt", 'w') as f:
        for i in range(cnt_all):
            if cnt_right_list[i]==0:
                wronglist_bin.append(result_bin[i])
                wronglist_true.append(result_right[i])
                wronglist_img.append(imglist[i])
                show_list.append(str(imglist[i])+"   ||| wrong result："+str(result_bin[i]))
                print(imglist[i] + "  " + str(result_bin[i]), file=f)
    time_cnt_hign1 = result_bin_0[-4].astype("uint8")
    time_cnt_hign2 = result_bin_0[-5].astype("uint8")
    time_cnt_hign3 = result_bin_0[-6].astype("uint8")
    time_cnt_hign4 = result_bin_0[-7].astype("uint8")
    time_cnt = time_cnt_hign1*2**24 + time_cnt_hign2*2**16 + time_cnt_hign3*2**8 + time_cnt_hign4*2**0
    time_cnt_hign1_first = result_bin_0[4].astype("uint8")
    time_cnt_hign2_first = result_bin_0[3].astype("uint8")
    time_cnt_hign3_first = result_bin_0[2].astype("uint8")
    time_cnt_hign4_first = result_bin_0[1].astype("uint8")
    time_cnt_first = time_cnt_hign1_first*2**24 + time_cnt_hign2_first*2**16 + time_cnt_hign3_first*2**8 + time_cnt_hign4_first*2**0
    time_ms = (time_cnt-time_cnt_first)*5*0.001*0.001/cnt_all
    fps = 1000/time_ms
    cal_power = fps * 14.99
    print("每张图处理时间为 " + str(time_ms) + " ms")
    print("处理帧率为 " + str(fps) + " 帧")
    print("等效算力为 " + str(cal_power) + " GOPS")
    print("*"*50)
