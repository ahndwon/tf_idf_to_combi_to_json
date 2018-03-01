# -*- coding: utf-8 -*-
from urllib.request import urlretrieve
import cv2
import json
from collections import defaultdict
import csv
import math
import heapq

import colormath
import numpy as np
import matplotlib.pyplot as plt

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

import os

from sklearn.cluster import MiniBatchKMeans, KMeans
from itertools import combinations

import matplotlib
from matplotlib import font_manager, rc
import pandas as pd
from firebase import firebase


class Combination:
    def __init__(self, keyword, image, color1, color2, color3):
        self.keyword = keyword
        self.image = image
        self.color1 = color1
        self.color2 = color2
        self.color3 = color3


def remove_blanks(a_list):
    new_list = []
    for item in a_list:
        if item != "" and item != "\n":
            new_list.append(item)
    return new_list


def get_ht_hash():
    temp_list = []

    # 윈도우 인코딩
    f = open('hueAndTone.csv', 'r', encoding='utf-8')
    i = 0
    while True:
        if i == 0:
            i = i + 1
            continue

        v = f.readline()
        if v == "":
            break

        s = v.split(',')

        s[6] = str(s[6])
        s[6] = str(s[6])[0:-1]

        if s[0] == "":
            break
        s = remove_blanks(s)
        temp_list.append(s)

    ht_num_list = []
    ht_rgb_list = []
    ht_color_list = []

    # 범위 1부터 : 첫줄 (헤더) 건너뜀
    for i in range(1, len(temp_list)):
        ht_num = temp_list[i][0]
        ht_color = temp_list[i][1]
        ht_rgb = []

        v = temp_list[i]

        r = v[4]
        g = v[5]
        b = v[6]

        rgb = sRGBColor(r, g, b)
        ht_rgb.append(rgb)

        ht_num_list.append(ht_num)
        ht_rgb_list.append(ht_rgb)
        ht_color_list.append(ht_color)

    ht_color_hash = dict(zip(ht_color_list, ht_rgb_list))
    ht_num_rgb_hash = dict(zip(ht_num_list, ht_rgb_list))

    f.close()

    return ht_num_rgb_hash


def get_combi_list():
    temp_list = []

    # 윈도우 인코딩
    f = open('num_combin.csv', 'r', encoding='utf-8')
    i = 0
    while True:
        if i == 0:
            i = i + 1
            continue

        v = f.readline()
        if v == "":
            break

        v = v.replace("\"", "")
        s = v.split(',')

        s[4] = str(s[4])[0:-1]

        if s[0] == "":
            break
        s = remove_blanks(s)
        temp_list.append(s)

    ht_num_list = []
    ht_rgb_list = []
    ht_color_list = []
    keyword_list = []
    image_list = []
    color1_list = []
    color2_list = []
    color3_list = []
    combi_list = []

    # 범위 1부터 : 첫줄 (헤더) 건너뜀
    for i in range(1, len(temp_list)):
        keyword = temp_list[i][0]
        image = temp_list[i][1]
        color1 = temp_list[i][2]
        color2 = temp_list[i][3]
        color3 = temp_list[i][4]

        combi = Combination(keyword, image, color1, color2, color3)

        keyword_list.append(keyword)
        image_list.append(image)
        color1_list.append(color1)
        color2_list.append(color2)
        color3_list.append(color3)

        combi_list.append(combi)

    ht_color_hash = dict(zip(ht_num_list, ht_color_list))
    ht_num_rgb_hash = dict(zip(ht_num_list, ht_rgb_list))

    f.close()

    return combi_list


def match_color(matching_color, ht_num_rgb_hash):
    result = -1
    for i in range(0, len(ht_num_rgb_hash.keys())):
        ht_num = list(ht_num_rgb_hash.keys())[i]
        for k in range(0, len(ht_num_rgb_hash.get(ht_num))):
            lab_matching_color = convert_color(matching_color, LabColor)
            lab_ht = convert_color(ht_num_rgb_hash.get(ht_num)[k], LabColor)
            if result == (-1):
                result = colormath.color_diff.delta_e_cie2000(lab_matching_color, lab_ht, 1, 1, 1)
                matched_rgb = ht_num_rgb_hash.get(ht_num)
                matched_num = ht_num

            else:
                if result > colormath.color_diff.delta_e_cie2000(lab_matching_color, lab_ht, 1, 1, 1):
                    result = colormath.color_diff.delta_e_cie2000(lab_matching_color, lab_ht, 1, 1, 1)
                    matched_rgb = ht_num_rgb_hash.get(ht_num)
                    matched_num = ht_num
                    # print("result[" + str(i) + "]: " + str(result))

    print("matched_rgb: ", matched_rgb)
    print("matched_num: ", matched_num)
    return matched_num


def match_combi(webtoon_combi):
    combi_list = get_combi_list()

    # print("webtoon_combi: ", webtoon_combi)
    # print("webtoon_combi: ", webtoon_combi[0])
    # print("webtoon_combi: ", webtoon_combi[1])
    # print("webtoon_combi: ", webtoon_combi[2])
    result_keyword = ""
    result_image = ""
    result_color1, result_color2, result_color3 = '', '', ''

    for i in range(0, len(combi_list)):
        # print("i: ", i)

        diff, keyword, image, f_color1, f_color2, f_color3 = match_combi_color(webtoon_combi, combi_list[i])
        if i == 0:
            # print(" i == 0")
            result = diff
            result_keyword = keyword
            result_image = image
            result_color1 = f_color1
            result_color2 = f_color2
            result_color3 = f_color3
        else:
            if result > diff:
                # print("#####FOUND!", i)
                # print("Found WORD : ", combi_list[i].keyword, combi_list[i].image)
                result = diff
                result_keyword = keyword
                result_image = image
                result_color1 = f_color1
                result_color2 = f_color2
                result_color3 = f_color3

    result_combi = Combination(result_keyword, result_image, result_color1, result_color2, result_color3)
    # print("result: ", result)
    # print("result_keyword: ", result_keyword)
    # print("result_image: ", result_image)
    # print("result_color1: ", result_color1)
    # print("result_color2: ", result_color2)
    # print("result_color3: ", result_color3)
    # print("result_combi: ", result_combi)
    # return result_image
    return result_combi


# 조합 색깔 매칭 / diff, 조합 반환
def match_combi_color(webtoon_combi, combi):
    ht_hash = get_ht_hash()

    lab_wc_color1 = convert_color(ht_hash.get(str(webtoon_combi[0]))[0], LabColor)
    lab_wc_color2 = convert_color(ht_hash.get(str(webtoon_combi[1]))[0], LabColor)
    lab_wc_color3 = convert_color(ht_hash.get(str(webtoon_combi[2]))[0], LabColor)
    lab_wc_list = [lab_wc_color1, lab_wc_color2, lab_wc_color3]

    lab_cb_color1 = convert_color(ht_hash.get(combi.color1)[0], LabColor)
    lab_cb_color2 = convert_color(ht_hash.get(combi.color2)[0], LabColor)
    lab_cb_color3 = convert_color(ht_hash.get(combi.color3)[0], LabColor)
    lab_cb_list = [lab_cb_color1, lab_cb_color2, lab_cb_color3]

    total_diff = 0
    for i in range(0, 3):

        diff = 0
        for j in range(0, 3):
            # print("for i: ", i)

            if j == 0:
                diff = colormath.color_diff.delta_e_cie2000(lab_wc_list[i], lab_cb_list[j], 1, 1, 1)
            else:
                if diff > colormath.color_diff.delta_e_cie2000(lab_wc_list[i], lab_cb_list[j], 1, 1, 1):
                    diff = colormath.color_diff.delta_e_cie2000(lab_wc_list[i], lab_cb_list[j], 1, 1, 1)

        total_diff = total_diff + diff

    keyword = combi.keyword
    image = combi.image

    # print("total_diff: ", total_diff)
    return total_diff, keyword, image, combi.color1, combi.color2, combi.color3


def find_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist


def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    top_ten_list = []

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        r = round(color[0])
        g = round(color[1])
        b = round(color[2])
        print(color[0])
        print("R: ", r, " G: ", g, "B: ", b)
        print(percent)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX
        top_ten_list.append([r, g, b, percent])

    # return the bar chart
    # print(top_ten_list)
    return bar, top_ten_list


def get_lists(episode):
    result_combi_list = []
    tf_idf_list = []

    # bar = np.zeros((50, 300, 3), dtype="uint8")
    ht_list = []
    for c in episode:
        ht_list.append(c.get('color'))

    for ti in episode:
        tf_idf_list.append(ti.get('tf-idf'))

    # 색, tf-idf 묶음
    tf_idf_dict = dict(zip(ht_list, tf_idf_list))
    # print('sorted:', sorted(tf_idf_dict.values(), reverse=True)[:5])
    # tf_idf_dict = dict(sorted(tf_idf_dict.values() reverse=True)[:5])
    # print('tf_idf_dict:', tf_idf_dict)
    top5 = heapq.nlargest(5, tf_idf_dict, key=tf_idf_dict.get)
    # print('len(top5):', len(top5))
    if len(top5) < 3:
        for i in range(0, 3-len(top5)):
            top5.append(top5[i])

    ht_list = top5
    # print('top5:', top5)
    # print("ht_list: ", ht_list)
    # print(heapq.nlargest(5, tf_idf_dict, key=tf_idf_dict.get))
    # print('ht_list:', ht_list)

    # 여기부터
    # ht_list = to_ht_tone(top_rgb_list)

    ht_combi_list = list_to_color_combi(ht_list)

    # print("ht_combi: ", ht_combi_list)
    combi_tf_idf_list = []

    for cb in ht_combi_list:
        result = match_combi(cb)
        result_combi_list.append(result)
        combi_tf_idf = get_tf_idf(cb, tf_idf_dict)
        combi_tf_idf_list.append(combi_tf_idf)
        #
        # print("match_combi_test(cb): ", result.keyword)
        # print("match_combi_test(cb)1: ", result.image)
        # print("match_combi_test(cb)2: ", result.color1)
        # print("match_combi_test(cb)3: ", result.color2)
        # print("match_combi_test(cb)4: ", result.color3)

    # print("result_combi_list: ", result_combi_list)

    # combi_tf_idf_dict = dict(zip(result_combi_list, combi_tf_idf_list))
    # print("combi_tf_idf_dict:", combi_tf_idf_dict)
    # plt.axis("off")
    # plt.imshow(bar)
    # plt.show()
    #
    # os.remove('tmp.jpg')
    return result_combi_list, combi_tf_idf_list


def get_tf_idf(combi, tf_idf_dict):
    tf_idf1 = tf_idf_dict.get(combi[0])
    tf_idf2 = tf_idf_dict.get(combi[1])
    tf_idf3 = tf_idf_dict.get(combi[2])
    return tf_idf1 + tf_idf2 + tf_idf3


def to_ht_tone(top_rgb_list):
    ht_hash = get_ht_hash()
    ht_list = []
    for rgb in top_rgb_list:
        ht_list.append(match_color(rgb, ht_hash))  # rgb -> ht_tone
    return ht_list


def list_to_color_combi(ht_list):
    return list(combinations(ht_list, 3))


def make_rgb_object(top_ten_list):
    top_ten_rgb = []
    for color in top_ten_list:
        top_ten_rgb.append(sRGBColor(color[0], color[1], color[2]))
    return top_ten_rgb


def data_to_json(episode, result_combi_list, tf_idf_list, xy_dict):
    color_info_list = list()
    # color_info_list.append(str(episode))
    for i in range(0, len(result_combi_list)):
        isUnique = True
        color_info = {
            'keyword': result_combi_list[i].keyword,
            'image': result_combi_list[i].image,
            'color1': result_combi_list[i].color1,
            'color2': result_combi_list[i].color2,
            'color3': result_combi_list[i].color3,
            'tf-idf': round(tf_idf_list[i], 2),
            'x': str(xy_dict.get(result_combi_list[i].image).get('x')),
            'y': str(xy_dict.get(result_combi_list[i].image).get('y'))
        }
        if len(color_info_list) is not 0:

            for j in range(1, len(color_info_list)):
                # print('color_info_list[j]:', color_info_list[j])
                # print('color_info_list[j]:', color_info_list[j].color1)
                # print('color_info_list[j]:', color_info_list[j].keyword)
                # print('color_info_list[j]:', color_info_list[j]['keyword'])
                if color_info_list[j].get('color1') == result_combi_list[i].color1 and color_info_list[j].get('color2') == result_combi_list[i].color2 and color_info_list[j].get('color3') == result_combi_list[i].color3:
                    isUnique = False
                    color_info = {
                        'keyword': result_combi_list[i].keyword,
                        'image': result_combi_list[i].image,
                        'color1': result_combi_list[i].color1,
                        'color2': result_combi_list[i].color2,
                        'color3': result_combi_list[i].color3,
                        'tf-idf': color_info_list[j].get('tf-idf') + round(tf_idf_list[i], 2),
                        'x': str(xy_dict.get(result_combi_list[i].image).get('x')),
                        'y': str(xy_dict.get(result_combi_list[i].image).get('y'))
                    }
                    color_info_list[j] = color_info
        if isUnique:
            color_info_list.append(color_info)
        # jsonString = json.dumps(webtoon_result, ensure_ascii=False)
        # print("color_info: ", color_info)
        # print("jsonStringType: ", type(jsonString))
    return color_info_list


def get_xy_dict():
    xy_csv = pd.read_csv("xy.csv")
    image_list = xy_csv['image']
    x_list = xy_csv['x']

    y_list = xy_csv['y']

    xy_list = []
    for i in range(0, 177):
        temp = {'x': x_list[i], 'y': y_list[i]}
        xy_list.append(temp)

    return dict(zip(image_list, xy_list))


def get_dict_from_csv():
    # 한글 폰트 깨짐 해결
    # temp = method1()
    # print(temp)
    # for key in temp:
    #     print(temp.get(key)[0])
    print(matplotlib.rcParams["font.family"])
    # 윈도우용 폰트 지정
    # font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()

    # 맥용 폰트 지정
    font_name = "AppleGothic"
    rc('font', family=font_name)

    json_csv = pd.read_csv("json4.csv")
    uniq_title = pd.unique(json_csv['title'])
    title = json_csv['title']
    episode_id = json_csv["episode_id"]
    uniq_episode_id = pd.unique(json_csv["episode_id"])
    ht_csv = pd.read_csv("hueAndTone.csv")
    ht_num = pd.unique(ht_csv["num"])

    # zero행렬 이용
    zero = np.zeros((len(uniq_episode_id), len(ht_num)))

    # zero행렬을 DataFrame으로 변환
    tf_dummy = pd.DataFrame(zero, index=uniq_episode_id, columns=ht_num)
    tf_idf_dummy = pd.DataFrame(zero, index=uniq_episode_id, columns=ht_num)
    # 더미행렬 -> 희소행렬
    # print(dummy)

    # print('title:', title)
    # title.to_csv('title.csv')

    for ht, e_id, ratio in zip(json_csv["hue & tone"], json_csv["episode_id"], json_csv["ratio"]):
        tf_dummy.ix[(e_id, ht)] += ratio

    # print(tf_dummy)
    idf_list = []
    ht_list = []
    title_list = []
    count = 0
    e_count = 0
    count_list = []
    e_list = []

    # title count
    start = 0
    count = 0
    for i in range(0, len(title)):
        count += 1

        if i != len(title) - 1 and episode_id[i] != episode_id[i + 1]:
            e_count += 1
            count_list.append(count)
            # print("[" + str(i) + "]: " + str(count))
            count = 0

        if i != len(title) - 1 and title[i] != title[i + 1]:
            e_list.append(e_count)
            title_list.append(title[i])
            # print("[e: " + str(title[i]) + "]: " + str(e_count))
            e_count = 0

        if i == len(title) - 1:
            # print('i == len(title)')
            e_count += 1
            count_list.append(count)
            e_list.append(e_count)
            title_list.append(title[i])

    # print("count_list: ", count_list)
    # print("e_list: ", e_list)
    # print("title_list:", title_list)
    # print('test:', dict(zip(title_list, e_list)))

    for ht in ht_num:
        ht_list.append(ht)

    for ht in ht_num:
        # idf_list.append(sum(tf_dummy[ht]))
        idf_list.append(sum(tf_dummy[ht]) / len(uniq_episode_id))

    idf_dict = dict(zip(ht_list, idf_list))
    # print("idf_dict: ", idf_dict)

    result_csv = open('result.csv', 'w', encoding='utf-8')
    result_csv.write(' ')

    for ht in json_csv["hue & tone"]:
        result_csv.write(str(ht) + ' ')
    result_csv.close()

    for ht, e_id, ratio in zip(json_csv["hue & tone"], json_csv["episode_id"], json_csv["ratio"]):
        if idf_dict[ht] != 0:
            tf_idf_dummy.ix[(e_id, ht)] += ratio / idf_dict[ht]


    # for e_id in json_csv["episode_id"]:

    # print(tf_idf_dummy)
    # tf_idf_dummy.to_csv('test.csv')
    color_tf_dict_list = []  #
    e_count = 0
    existing_ep = []
    for e_id in uniq_episode_id:
        temp_list = []
        for ht in ht_num:
            if tf_idf_dummy[ht][e_id] != 0:
                # print(e_id, ht, tf_idf_dummy[ht][e_id])
                temp_list.append({'color': ht, 'tf-idf': math.log(tf_idf_dummy[ht][e_id])})
        #         e_count += 1
        # existing_ep.append(e_count)
        # e_count = 0


        # print("temp_list:", temp_list)
        color_tf_dict_list.append(temp_list)

    # print("list2:", list2)
    c_temp_list = []
    e_temp_list = []

    ec = 0
    j = 0
    # print('e_list:', e_list)
    # print('existing_ep:', existing_ep)
    # print('len(color_tf_dict_list): ', len(color_tf_dict_list), len(uniq_episode_id), sum(e_list))
    # print('existing_ep:', sum(existing_ep))
################################ 버그 : e_list(6452)랑 uniq_episode_id(6170)랑 길이가 다름

    ec_dict_list = []
    for i in range(len(color_tf_dict_list)):
        j += 1
        c_temp_list.append(color_tf_dict_list[i])
        e_temp_list.append(uniq_episode_id[i])
        # print('[' + str(i) + ']' + str(uniq_episode_id[i]))
        if (j - 1) == e_list[ec] - 1:
            # print('위:', uniq_title[ec])
            temp_dict = dict(zip(e_temp_list, c_temp_list))
            ec_dict_list.append(temp_dict)
            c_temp_list.clear()
            e_temp_list.clear()
            ec += 1
            j = 0
    # print("ec_dict:", ec_dict_list)
    # print("ec_dict_list:", ec_dict_list)

    title_ep_dict = dict(zip(title_list, ec_dict_list))
    # print('title_ep_dict:', title_ep_dict)

    # with open('test.txt', 'w') as file:
    #     file.write(json.dumps(title_ep_dict))

    # np.save('test.npy', title_ep_dict)
    # Term Document Matrix형식으로 변경
    TDM = tf_dummy.T
    # TDM = tf_idf_dummy.T
    # print(TDM)
    # word_counter = TDM.sum(axis=1)  # 행 단위 합계
    # print("word_counter: ", word_counter)

    # 빈도수 시각화하기
    # word_counter.plot(kind='barh', title='voca counter')

    # 내림차순 정렬
    # word_counter.sort_values().plot(kind='barh', title='voca counter')
    # plt.show()
    # print(title_ep_dict)
    return title_ep_dict


if __name__ == '__main__':
    SECRET = 'xTaafnucUDwHmWeTAcikP7O3uThFeTSFGWjY28iH'
    DSN = 'https://first-webtoon-visualization.firebaseio.com/'
    authentication = firebase.FirebaseAuthentication(SECRET, None, False, False)  # image_kmeans(e_dict.get(32022))
    firebase = firebase.FirebaseApplication(DSN, authentication)

    # firebase.put('/result2', 'test', 'data')
    title_ep_dict = get_dict_from_csv()
    #
    color_db = firebase.get('/result2', None)
    xy_dict = get_xy_dict()
    episodes = []
    colors = []
    keys = sorted(title_ep_dict.keys())
    for t in keys[keys.index('HeavensSoul'):]:
        if (color_db is not None) and (color_db.get(t) is None):
            print("title:", t)
            for e in title_ep_dict.get(t):
                print('episode:', e)
                # print('episode:', e)
                # print(e_dict.get(e))
                result_list, idf_list = get_lists(title_ep_dict.get(t).get(e))
                # print('result_list:', result_list)
                # print('ratio_list:', idf_list)
                json = data_to_json(e, result_list, idf_list, xy_dict)
                result = {
                    str(e): json
                }
                colors.append(result)
                # episodes.append(e.key())
            if (color_db is not None) and (color_db.get(t) is None):
                firebase.put('/result2', t, colors)
            else:
                print(t + 'already exists')
            colors.clear()

    # t = 'dings'
    # print("title:", t)
    # for e in title_ep_dict.get(t):
    #     print('episode:', e)
    #     # print(e_dict.get(e))
    #     result_list, idf_list = get_lists(title_ep_dict.get(t).get(e))
    #     # print('result_list:', result_list)
    #     # print('ratio_list:', idf_list)
    #     json = data_to_json(e, result_list, idf_list, xy_dict)
    #     result = {
    #         str(e): json
    #     }
    #     colors.append(result)
    #     print(json)
    #     # episodes.append(e.key())
    # if (color_db is not None) and (color_db.get(t) is None):
    #     firebase.put('/analysis', t, colors)
    #
    # else:
    #     print(t + 'already exists')
    # colors.clear()
    #
    print("finished")


