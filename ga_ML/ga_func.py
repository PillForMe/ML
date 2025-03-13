import pandas as pd
import numpy as np
import random
import copy
import time
import re

association_analysis_data = {
	1: [6,27,2,9,28,11,14,8,21,3,5,19,26],
	2: [6,27,9,28,11,8,13,22,21,1,3,5,19,16,20,26],
	3: [6,27,2,9,28,11,8,13,21,1,5,19,20,26],
	4: [],
	5: [6,27,2,18,9,28,11,14,8,22,21,1,3,19,16,20,26],
	6: [27,2,9,28,11,14,8,21,1,3,5,19,16,26],
	7: [],
	8: [27,2,18,9,28,11,13,22,21,1,3,5,19,16,20,26],
	9: [6,27,12,2,28,11,14,8,13,22,21,1,3,5,19,16,20,26],
	10: [],
	11: [6,27,2,9,28,8,13,22,21,1,3,5,19,26],
	12: [9,21,26],
	13: [27,2,9,11,8,22,21,3,19],
	14: [],
	15: [],
	16: [6,2,9,3,5,19],
	17: [],
	18: [2,8,5,20],
	19: [6,27,2,9,28,11,8,13,22,21,1,3,5,16,20,26],
	20: [2,18,8,5],
	21: [6,27,12,2,9,28,11,8,13,22,21,1,3,5,19,26],
	22: [9,11,8,13,21],
	23: [],
	24: [],
	25: [],
	26: [6,27,12,2,9,28,11,8,22,21,1,3,5,19,20],
	27: [6,2,9,28,11,8,13,22,21,1,3,5,19,26],
	28: [6,27,2,9,11,8,21,1,3,5,19,16,26]
    }

# 건강 고민 정보
health_concern_data = {
    '피로감':1,
    '눈건강':2,
    '피부건강':3,
    '체지방':4,
    '혈관혈액순환':5,
    '간건강':6,
    '장건강':7,
    '스트레스&수면':8,
    '면역기능':9,
    '혈중콜레스테롤':10,
    '뼈건강':11,
    '노화&항산화':12,
    '노화&향산화':12, # 원데이터에 오타 있어서 추가함
    '여성건강':13,
    '소화&위식도건강':14,
    '남성건강':15,
    '혈압':16,
    '운동능력&근육량':17,
    '두뇌활동':18,
    '혈당':19,
    '혈중중성지방':20,
    '치아&잇몸':21,
    '치아잇몸&잇몸건강':21, # 원데이터에 오타 있어서 추가함
    '임산부&태아건강':22,
    '탈모&손톱건강':23,
    '관절건강':24,
    '여성갱년기':25,
    '호흡기건강':26,
    '갑상선건강':27,
    '빈혈':28
}


health_concern_with_ingredient_data = {
    '피로감': [15], '눈건강': [7], '피부건강': [15], '체지방': [13], '혈관혈액순환': [2], '간건강': [11], '장건강': [25], '스트레스&수면': [23], '면역기능': [25], '혈중콜레스테롤': [7], '뼈건강': [25], '노화&향산화': [9], '여성건강': [23], '소화&위식도건강': [25], '남성건강': [25], '혈압': [22], '운동능력&근육량': [11], '두뇌활동': [2], '혈당': [31], '혈중중성지방': [2], '치아&잇몸': [18, 9], '임산부&태아건강': [23], '탈모&손톱건강': [13], '관절건강': [6], '여성갱년기': [], '호흡기건강': [9], '갑상선건강': [28], '빈혈': [16]
    }

nutrient_raw_data = pd.read_csv("./최종_영양제_함량정보(to지현).csv", encoding='utf-8') ## 👨🏻‍💻 경로 변경
# '영양함량' column drop
nutrient_raw_data.drop(columns=['영양함량'], inplace=True)
# NaN 값은 0으로 대체
nutrient_raw_data = nutrient_raw_data.fillna(0)

item_health_grouped_df_path = "./item_health_grouped_df.csv" ## 👨🏻‍💻 경로 변경
limitation_df_path = "./성분별한계량.csv" ## 👨🏻‍💻 경로 변경
ocr_df_path = "./ocr_result1_비교용.csv" ## 👨🏻‍💻 경로 변경


# random.seed(42) #
# np.random.seed(42) #

def ga(user_1, nutrient_data, item_health_grouped_df_path, limitation_df_path, ocr_df_path, generation_num, initial_population_num):
    # 관련된 카테고리 추출하기
    associated_category_list = []
    for category in user_1['preference_category']:
        associated_category_list.append(category)
        associated_category_list.extend(association_analysis_data[category])
    associated_category_list = list(set(associated_category_list))

    rawdata_nutrient_name = nutrient_data['브랜드명_제품명'].tolist()
    category_raw_data = pd.read_csv(item_health_grouped_df_path, encoding='utf-8')
    # 기존의 건강고민정보를 숫자로 변경
    row_idx = 0
    for health_concern in category_raw_data['건강 고민 정보']:
        health_concern_list = health_concern.split(',')
        health_concern_as_int_list = []
        for each_health_concern in health_concern_list:
            health_concern_num = health_concern_data[each_health_concern]
            health_concern_as_int_list.append(health_concern_num)
        # 기존에 글자(['남성건강', '운동능력&근육량'])를 숫자([15, 17])로 변경함
        category_raw_data['건강 고민 정보'][row_idx] = health_concern_as_int_list
        row_idx += 1
    category_raw_data.rename(columns={'건강 고민 정보': '카테고리'}, inplace=True)

    # 기존 raw_data와 영양제 idx 일치시키기
    name_and_category_list = list(zip(category_raw_data['브랜드명_제품명'], category_raw_data['카테고리']))
    rawdata_nutrient_name_and_category = []
    for nutrient in rawdata_nutrient_name:
        for item in name_and_category_list:
            if nutrient == item[0]:
                rawdata_nutrient_name_and_category.append(item)
                break
    nutrient_category = [item[1] for item in rawdata_nutrient_name_and_category]
    item_idx_of_custom_category = []
    for index, sublist in enumerate(nutrient_category):
        if any(item in associated_category_list for item in sublist):
            item_idx_of_custom_category.append(index)

    # 유효한 인덱스만 선택
    valid_indices = [idx for idx in item_idx_of_custom_category if idx in nutrient_data.index]

    nutrient_data = nutrient_data.loc[valid_indices]
    nutrient_data.reset_index(drop=True, inplace=True)

    nutrient_list = nutrient_data.columns.tolist()
    # ✅ 일부만 전처리 진행
    wanted_column = ['탄수화물', '차전자피식이섬유', '식이섬유', '귀리식이섬유', '난소화성말토덱스트린(식이섬유)', '오메가3(EPA+DHA)', '단백질', 'L-메티오닌', 'MMSC(메틸메티오닌설포늄염화물)', '셀레노메티오닌', 'L-시스테인', '비타민A', '레티놀팔미테이트(비타민A)', '베타카로틴(비타민A)', '레티놀아세테이트(비타민A)', '비타민D', '비타민D3', '비타민D2', '비타민E', '비타민E(디알파토코페릴숙시네이트)', '비타민E(디엘알파토코페릴아세테이트)', '비타민E(디알파토코페릴아세테이트)', '비타민E(디알파코페릴숙시네이트)', '비타민E(디알파토코페롤)', '비타민E(디알팤페릴아세테이트)', '비타민E(디엘알파토코페릴숙시네이트)', '비타민K', '비타민K2', '비타민K1','비타민C', '비타민C(퀄리?-C)', '비타민C(퀄리-C)','비타민B1', '티아민질산염(B1)', '티아민염산염(B1)', '벤포티아민(B1)', '푸르설티아민(B1)','비타민B2', '리보플라빈포스페이트(B2)', '비타민(B2)', '리보플라빈부티레이트(B2)','나이아신(비타민B3)', '비타민B3(나이아신아마이드)', '비타민B3(나이아신)', '비타민B3(나이아신마이드)','판토텐산(B5)', '판토텐산(비타민B5)', '펜토텐산(B5)', '핀토텐산(B5)', '판토텐산칼슘', '비타민B6', '피리독신염산염(B6)', '피리독살포스페이트(B6)', '비오틴', '엽산', '엽산(쿼트라폴릭)', '메틸엽산', '폴레이트(엽산)', '비타민B12', '시아노코발라민(B12)', '메코발라민(B12)', '히드록소코발라민(B12)', '시안코발라민(B12)', '하드록소코발라민(B12)', '해조칼슘', '칼슘', '인', '나트륨', '염소(클로라이드)', '칼륨', '마그네슘', '철', '페리친(철)', '아연', '구리', '망간', '요오드', '셀레늄(셀렌)', '셀레늄','셀렌산나트륨', '몰리브덴', '몰리브덴아미노산킬레이트', '크롬', '크롬아미노산킬레이트', '크롬피콜리네이트(피콜린산크롬)', '루테인', '루테인지아잔틴', '루테인(루테맥스)', '루테인(플로라지엘오)', '지아잔틴']
    wanted_df = nutrient_data.loc[:, wanted_column]

    # ✅ 전처리 (1): 단위 전처리
    # 특이 단위를 제외하고 모두 'μg'로 단위 변경
    wanted_column1_for_prepro1 = ['탄수화물', '차전자피식이섬유', '식이섬유', '귀리식이섬유', '난소화성말토덱스트린(식이섬유)', '오메가3(EPA+DHA)', '단백질', 'L-메티오닌', 'MMSC(메틸메티오닌설포늄염화물)', '셀레노메티오닌', 'L-시스테인', '비타민D', '비타민D3', '비타민D2', '비타민K', '비타민K2', '비타민K1','비타민C', '비타민C(퀄리?-C)', '비타민C(퀄리-C)','비타민B1', '티아민질산염(B1)', '티아민염산염(B1)', '벤포티아민(B1)', '푸르설티아민(B1)','비타민B2', '리보플라빈포스페이트(B2)', '비타민(B2)', '리보플라빈부티레이트(B2)','판토텐산(B5)', '판토텐산(비타민B5)', '펜토텐산(B5)', '핀토텐산(B5)', '판토텐산칼슘', '비타민B6', '피리독신염산염(B6)', '피리독살포스페이트(B6)', '비오틴', '엽산', '엽산(쿼트라폴릭)', '메틸엽산', '폴레이트(엽산)', '비타민B12', '시아노코발라민(B12)', '메코발라민(B12)', '히드록소코발라민(B12)', '시안코발라민(B12)', '하드록소코발라민(B12)', '해조칼슘', '칼슘', '인', '나트륨', '염소(클로라이드)', '칼륨', '마그네슘', '철', '페리친(철)', '아연', '구리', '망간', '요오드', '셀레늄(셀렌)', '셀레늄','셀렌산나트륨', '몰리브덴', '몰리브덴아미노산킬레이트', '크롬', '크롬아미노산킬레이트', '크롬피콜리네이트(피콜린산크롬)', '루테인', '루테인지아잔틴', '루테인(루테맥스)', '루테인(플로라지엘오)', '지아잔틴']

    for col_name in wanted_column1_for_prepro1:
        for row_idx in range(wanted_df.shape[0]):
            value = wanted_df[col_name][row_idx]
            if value == 0 :
                continue
            # 숫자 부분 추출
            number_part = float(re.findall(r'[\d\.]+', value)[0])
            # 문자 부분 추출
            text_part = re.findall(r'[^\d\.]+', value)[0].strip().replace(" ", "")
            # 모두 μg으로 변경하기
            if text_part == 'μg':
                changed_value = number_part
            if text_part == 'mg':
                changed_value = number_part * 1000
            if text_part == 'g':
                changed_value = number_part * 1000000
            wanted_df[col_name][row_idx] = changed_value


    for col_name in ['비타민A', '레티놀팔미테이트(비타민A)', '레티놀아세테이트(비타민A)']:
        for row_idx in range(wanted_df.shape[0]):
            value = wanted_df[col_name][row_idx]
            if value == 0 :
                continue
            # 숫자 부분 추출
            number_part = float(re.findall(r'[\d\.]+', value)[0])
            # 문자 부분 추출
            text_part = re.findall(r'[^\d\.]+', value)[0].strip().replace(" ", "")
            wanted_df[col_name][row_idx] = number_part


    for row_idx in range(wanted_df.shape[0]):
        value = wanted_df['베타카로틴(비타민A)'][row_idx]
        if value == 0 :
            continue
        # 숫자 부분 추출
        number_part = float(re.findall(r'[\d\.]+', value)[0])
        # 문자 부분 추출
        text_part = re.findall(r'[^\d\.]+', value)[0].strip().replace(" ", "")
        # 우선 g과 mg을 μg으로 변경
        if text_part == 'μg':
            changed_value = number_part
        if text_part == 'mg':
            changed_value = number_part * 1000
        if text_part == 'g':
            changed_value = number_part * 1000000
        # μg을 μgRAE으로 변경 (베타카로틴의 경우, 1 μgRAE는 약 0.5 μg의 베타카로틴에 해당)
        changed_value = changed_value * 2
        wanted_df['베타카로틴(비타민A)'][row_idx] = changed_value


    for col_name in ['비타민E', '비타민E(디알파토코페릴숙시네이트)', '비타민E(디엘알파토코페릴아세테이트)', '비타민E(디알파토코페릴아세테이트)', '비타민E(디알파코페릴숙시네이트)', '비타민E(디알파토코페롤)', '비타민E(디알팤페릴아세테이트)', '비타민E(디엘알파토코페릴숙시네이트)']:
        for row_idx in range(wanted_df.shape[0]):
            value = wanted_df[col_name][row_idx]
            if value == 0 :
                continue
            # 숫자 부분 추출
            number_part = float(re.findall(r'[\d\.]+', value)[0])
            # 문자 부분 추출
            text_part = re.findall(r'[^\d\.]+', value)[0].strip().replace(" ", "")
            wanted_df[col_name][row_idx] = number_part


    for col_name in ['나이아신(비타민B3)', '비타민B3(나이아신아마이드)', '비타민B3(나이아신)', '비타민B3(나이아신마이드)']:
        for row_idx in range(wanted_df.shape[0]):
            value = wanted_df[col_name][row_idx]
            if value == 0 :
                continue
            # 숫자 부분 추출
            number_part = float(re.findall(r'[\d\.]+', value)[0])
            # 문자 부분 추출
            text_part = re.findall(r'[^\d\.]+', value)[0].strip().replace(" ", "")
            wanted_df[col_name][row_idx] = number_part

    # ✅ 전처리 (2): 동일한 성분은 모두 더해주기
    # 특이 단위(아래 3종류) 제외하고 모든 단위는 'μg'으로 통일되어 있음
    # 모든 종류의 비타민 A => μg RAE
    # 모든 종류의 비타민 E => mg α-TE
    # 모든 종류의 비타민 B3 => mgNE

    preprocessed_df = wanted_df.copy()
    preprocessed_df['탄수화물 (μg)'] = wanted_df.iloc[:, 0]
    preprocessed_df = preprocessed_df.iloc[:,1:] # 위에 우변의 개수임
    preprocessed_df['식이섬유 (μg)'] = wanted_df.iloc[:, 1]+wanted_df.iloc[:, 2]+wanted_df.iloc[:, 3]+wanted_df.iloc[:, 4]
    preprocessed_df = preprocessed_df.iloc[:,4:]
    preprocessed_df['오메가3(EPA+DHA) (μg)'] = wanted_df.iloc[:, 5]
    preprocessed_df = preprocessed_df.iloc[:,1:]
    preprocessed_df['단백질 (μg)'] = wanted_df.iloc[:, 6]
    preprocessed_df = preprocessed_df.iloc[:,1:]
    preprocessed_df['메티오닌+시스테인 (μg)'] = wanted_df.iloc[:, 7]+wanted_df.iloc[:, 8]+wanted_df.iloc[:, 9]+wanted_df.iloc[:, 10]
    preprocessed_df = preprocessed_df.iloc[:,4:]
    preprocessed_df['비타민A (μg RAE)'] = wanted_df.iloc[:, 11]+wanted_df.iloc[:, 12]+wanted_df.iloc[:, 13]+wanted_df.iloc[:, 14]
    preprocessed_df = preprocessed_df.iloc[:,4:]
    preprocessed_df['비타민D (μg)'] = wanted_df.iloc[:, 15]+wanted_df.iloc[:, 16]+wanted_df.iloc[:, 17]
    preprocessed_df = preprocessed_df.iloc[:,3:]
    preprocessed_df['비타민E (mg α-TE)'] = wanted_df.iloc[:, 18]+wanted_df.iloc[:, 19]+wanted_df.iloc[:, 20]+wanted_df.iloc[:, 21]+wanted_df.iloc[:, 22]+wanted_df.iloc[:, 23]+wanted_df.iloc[:, 24]+wanted_df.iloc[:, 25]
    preprocessed_df = preprocessed_df.iloc[:,8:]
    preprocessed_df['비타민K (μg)'] = wanted_df.iloc[:, 26]+wanted_df.iloc[:, 27]+wanted_df.iloc[:, 28]
    preprocessed_df = preprocessed_df.iloc[:,3:]
    preprocessed_df['비타민C (μg)'] = wanted_df.iloc[:, 29]+wanted_df.iloc[:, 30]+wanted_df.iloc[:, 31]
    preprocessed_df = preprocessed_df.iloc[:,3:]
    preprocessed_df['비타민B1 (μg)'] = wanted_df.iloc[:, 32]+wanted_df.iloc[:, 33]+wanted_df.iloc[:, 34]+wanted_df.iloc[:, 35]+wanted_df.iloc[:, 36]
    preprocessed_df = preprocessed_df.iloc[:,5:]
    preprocessed_df['비타민B2 (μg)'] = wanted_df.iloc[:, 37]+wanted_df.iloc[:, 38]+wanted_df.iloc[:, 39]+wanted_df.iloc[:, 40]
    preprocessed_df = preprocessed_df.iloc[:,4:]
    preprocessed_df['비타민B3 (mg NE)'] = wanted_df.iloc[:, 41]+wanted_df.iloc[:, 42]+wanted_df.iloc[:, 43]+wanted_df.iloc[:, 44]
    preprocessed_df = preprocessed_df.iloc[:,4:]
    preprocessed_df['비타민B5 (μg)'] = wanted_df.iloc[:, 45]+wanted_df.iloc[:, 46]+wanted_df.iloc[:, 47]+wanted_df.iloc[:, 48]+wanted_df.iloc[:, 49]
    preprocessed_df = preprocessed_df.iloc[:,5:]
    preprocessed_df['비타민B6 (μg)'] = wanted_df.iloc[:, 50]+wanted_df.iloc[:, 51]+wanted_df.iloc[:, 52]
    preprocessed_df = preprocessed_df.iloc[:,3:]
    preprocessed_df['비타민B7 (μg)'] = wanted_df.iloc[:, 53]
    preprocessed_df = preprocessed_df.iloc[:,1:]
    preprocessed_df['비타민B9 (μg)'] = wanted_df.iloc[:, 54]+wanted_df.iloc[:, 55]+wanted_df.iloc[:, 56]+wanted_df.iloc[:, 57]
    preprocessed_df = preprocessed_df.iloc[:,4:]
    preprocessed_df['비타민B12 (μg)'] = wanted_df.iloc[:, 58]+wanted_df.iloc[:, 59]+wanted_df.iloc[:, 60]+wanted_df.iloc[:, 61]+wanted_df.iloc[:, 62]+wanted_df.iloc[:, 63]
    preprocessed_df = preprocessed_df.iloc[:,6:]
    preprocessed_df['칼슘 (μg)'] = wanted_df.iloc[:, 64]+wanted_df.iloc[:, 65]
    preprocessed_df = preprocessed_df.iloc[:,2:]
    preprocessed_df['인 (μg)'] = wanted_df.iloc[:, 66]
    preprocessed_df = preprocessed_df.iloc[:,1:]
    preprocessed_df['나트륨 (μg)'] = wanted_df.iloc[:, 67]
    preprocessed_df = preprocessed_df.iloc[:,1:]
    preprocessed_df['염소 (μg)'] = wanted_df.iloc[:, 68]
    preprocessed_df = preprocessed_df.iloc[:,1:]
    preprocessed_df['칼륨 (μg)'] = wanted_df.iloc[:, 69]
    preprocessed_df = preprocessed_df.iloc[:,1:]
    preprocessed_df['마그네슘 (μg)'] = wanted_df.iloc[:, 70]
    preprocessed_df = preprocessed_df.iloc[:,1:]
    preprocessed_df['철 (μg)'] = wanted_df.iloc[:, 71]+wanted_df.iloc[:, 72]
    preprocessed_df = preprocessed_df.iloc[:,2:]
    preprocessed_df['아연 (μg)'] = wanted_df.iloc[:, 73]
    preprocessed_df = preprocessed_df.iloc[:,1:]
    preprocessed_df['구리 (μg)'] = wanted_df.iloc[:, 74]
    preprocessed_df = preprocessed_df.iloc[:,1:]
    preprocessed_df['망간 (μg)'] = wanted_df.iloc[:, 75]
    preprocessed_df = preprocessed_df.iloc[:,1:]
    preprocessed_df['요오드 (μg)'] = wanted_df.iloc[:, 76]
    preprocessed_df = preprocessed_df.iloc[:,1:]
    preprocessed_df['셀레늄 (μg)'] = wanted_df.iloc[:, 77]+wanted_df.iloc[:, 78]+wanted_df.iloc[:, 79]
    preprocessed_df = preprocessed_df.iloc[:,3:]
    preprocessed_df['몰리브덴 (μg)'] = wanted_df.iloc[:, 80]+wanted_df.iloc[:, 81]
    preprocessed_df = preprocessed_df.iloc[:,2:]
    preprocessed_df['크롬 (μg)'] = wanted_df.iloc[:, 82]+wanted_df.iloc[:, 83]+wanted_df.iloc[:, 84]
    preprocessed_df = preprocessed_df.iloc[:,3:]
    preprocessed_df['루테인 (μg)'] = wanted_df.iloc[:, 85]+wanted_df.iloc[:, 86]+wanted_df.iloc[:, 87]+wanted_df.iloc[:, 88]
    preprocessed_df = preprocessed_df.iloc[:,4:]
    preprocessed_df['지아잔틴 (μg)'] = wanted_df.iloc[:, 89]
    preprocessed_df = preprocessed_df.iloc[:,1:]

    rawdata_nutrient_name = nutrient_data['브랜드명_제품명'].tolist()

    # category_raw_data의 영양제 수는 732개이지만, 총 영양제(711)를 모두 포함하고 있음
    category_raw_data = pd.read_csv(item_health_grouped_df_path, encoding='utf-8')
    # 기존의 건강고민정보를 숫자로 변경
    row_idx = 0
    for health_concern in category_raw_data['건강 고민 정보']:
        health_concern_list = health_concern.split(',')
        health_concern_as_int_list = []
        for each_health_concern in health_concern_list:
            health_concern_num = health_concern_data[each_health_concern]
            health_concern_as_int_list.append(health_concern_num)
        # 기존에 글자(['남성건강', '운동능력&근육량'])를 숫자([15, 17])로 변경함
        category_raw_data['건강 고민 정보'][row_idx] = health_concern_as_int_list
        row_idx += 1
    category_raw_data.rename(columns={'건강 고민 정보': '카테고리'}, inplace=True)

    # 기존 raw_data와 영양제 idx 일치시키기
    name_and_category_list = list(zip(category_raw_data['브랜드명_제품명'], category_raw_data['카테고리']))
    rawdata_nutrient_name_and_category = []
    for nutrient in rawdata_nutrient_name:
        for item in name_and_category_list:
            if nutrient == item[0]:
                rawdata_nutrient_name_and_category.append(item)
                break
    nutrient_category = [item[1] for item in rawdata_nutrient_name_and_category]

    # [카테고리]: 13(여성건강), 15(남성건강), 22(임산부&태아), 25(여성 갱년기)
    category_raw_data['성별'] = 2

    for idx in range(category_raw_data.shape[0]):
        category_list = category_raw_data.loc[idx, '카테고리']
        brand_name = category_raw_data.loc[idx, '브랜드명_제품명']
        if len(set([13,22,25]) & set(category_list)) > 0:
            category_raw_data.loc[idx, '성별'] = 0
        if 15 in category_list:
            category_raw_data.loc[idx, '성별'] = 1

        if '여성' in brand_name:
            category_raw_data.loc[idx, '성별'] = 0
        if '우먼' in brand_name:
            category_raw_data.loc[idx, '성별'] = 0

        if '남성' in brand_name:
            category_raw_data.loc[idx, '성별'] = 1
        if '맨' in brand_name:
            category_raw_data.loc[idx, '성별'] = 1

    # 🌷 성별 리스트 저장하기
    name_and_sex_list = list(zip(category_raw_data['브랜드명_제품명'], category_raw_data['성별']))
    rawdata_nutrient_name_and_sex = []
    for nutrient in rawdata_nutrient_name:
        for item in name_and_sex_list:
            if nutrient == item[0]:
                rawdata_nutrient_name_and_sex.append(item)
                break
    nutrient_gender_list = [item[1] for item in rawdata_nutrient_name_and_sex]

    limitataion_for_each_ingredient_data = pd.read_csv(limitation_df_path, encoding='cp949')
    # 75는 75~150살로 기입해두기
    limitataion_for_each_ingredient_data.loc[limitataion_for_each_ingredient_data['연령']=='75', '연령'] = '75~150'

    # ✔ 위 데이터에서 '상한섭취량'이 '무한'인 것은 'inf'로 바꿔줌
    limitataion_for_each_ingredient_data.loc[limitataion_for_each_ingredient_data['상한섭취량']=='무한', '상한섭취량'] = np.inf

    # ✔ '대상' 수정 (해당사항 없음=0, 임신부=1, 수유부=2)
    limitataion_for_each_ingredient_data.loc[limitataion_for_each_ingredient_data['대상']=='해당사항 없음', '대상'] = 0
    limitataion_for_each_ingredient_data.loc[limitataion_for_each_ingredient_data['대상']=='임신부', '대상'] = 1
    limitataion_for_each_ingredient_data.loc[limitataion_for_each_ingredient_data['대상']=='수유부', '대상'] = 2

    # ✔ '성별' 수정 (여자=0, 남자=1)
    limitataion_for_each_ingredient_data.loc[limitataion_for_each_ingredient_data['성별']=='여', '성별'] = 0
    limitataion_for_each_ingredient_data.loc[limitataion_for_each_ingredient_data['성별']=='남', '성별'] = 1

    # ✔ '연령' 수정
    def process_age(age_str):
        if isinstance(age_str, str) and '영아' in age_str:
            return age_str
        elif isinstance(age_str, str) and '~' in age_str:
            return list(range(int(age_str.split('~')[0]), int(age_str.split('~')[1]) + 1))
        else:
            return age_str
    limitataion_for_each_ingredient_data['연령']=limitataion_for_each_ingredient_data['연령'].apply(process_age)
    limitataion_for_each_ingredient_data.loc[limitataion_for_each_ingredient_data['연령']=='영아 (0~5개월)', '연령'] = 0.5
    limitataion_for_each_ingredient_data.loc[limitataion_for_each_ingredient_data['연령']=='영아 (6~11개월)', '연령'] = 0.9

    # 🌷 상한 섭취량
    nutrient_intake_limit = []

    # 🌷 평균 필요량
    nutrient_must_intake = []

    user_gender = user_1['gender']
    user_age = user_1['age']
    user_condition = user_1['condition']

    if user_1['age'] == '영아 (0~5개월)' or user_1['age'] == '영아 (6~11개월)':
        # 영아인 경우
        if user_1['age'] == '영아 (0~5개월)':
            user_age = 0.5
        else:
            user_age = 0.9
        user_gender = '무관'
        for component in preprocessed_df.columns.tolist():
            filtered_df = limitataion_for_each_ingredient_data[(limitataion_for_each_ingredient_data['영양제'] == component) & (limitataion_for_each_ingredient_data['대상'] == user_condition) & (limitataion_for_each_ingredient_data['성별'] == user_gender) & (limitataion_for_each_ingredient_data['연령'] == user_age)]
            upper_value = filtered_df['상한섭취량'].iloc[0]
            lower_value = filtered_df['평균필요량'].iloc[0]
            nutrient_intake_limit.append(float(upper_value))
            nutrient_must_intake.append(lower_value)
    else:
        for component in preprocessed_df.columns.tolist():
            if user_condition == 0:
                filtered_df = limitataion_for_each_ingredient_data[(limitataion_for_each_ingredient_data['영양제'] == component) & (limitataion_for_each_ingredient_data['대상'] == user_condition) & (limitataion_for_each_ingredient_data['성별'] == user_gender) & (limitataion_for_each_ingredient_data['연령'].apply(lambda x: isinstance(x, list) and user_age in x))]
                upper_value = filtered_df['상한섭취량'].iloc[0]
                lower_value = filtered_df['평균필요량'].iloc[0]
                nutrient_intake_limit.append(float(upper_value))
                nutrient_must_intake.append(lower_value)

            # 🌷 '임신부 or 수유부'일 경우, '평균필요량'의 값을 + 해줘야 함
            else:
                # '나트륨', '염소'는 '충분섭취량' 임신부/수유부가 +가 아니라 해당 그 값임
                if component == '나트륨 (μg)' or component == '염소 (μg)':
                    filtered_df = limitataion_for_each_ingredient_data[(limitataion_for_each_ingredient_data['영양제'] == component) & (limitataion_for_each_ingredient_data['대상'] == user_condition) & (limitataion_for_each_ingredient_data['성별'] == user_gender) & (limitataion_for_each_ingredient_data['연령'].isna())]
                    upper_value = filtered_df['상한섭취량'].iloc[0]
                    lower_value = filtered_df['평균필요량'].iloc[0]
                else:
                    filtered_df = limitataion_for_each_ingredient_data[(limitataion_for_each_ingredient_data['영양제'] == component) & (limitataion_for_each_ingredient_data['대상'] == 0) & (limitataion_for_each_ingredient_data['성별'] == user_gender) & (limitataion_for_each_ingredient_data['연령'].apply(lambda x: isinstance(x, list) and user_age in x))]
                    filtered_df2 = limitataion_for_each_ingredient_data[(limitataion_for_each_ingredient_data['영양제'] == component) & (limitataion_for_each_ingredient_data['대상'] == user_condition) & (limitataion_for_each_ingredient_data['성별'] == user_gender) & (limitataion_for_each_ingredient_data['연령'].isna())]
                    upper_value = filtered_df['상한섭취량'].iloc[0]
                    lower_value1 = filtered_df['평균필요량'].iloc[0]
                    lower_value2 = filtered_df2['평균필요량'].iloc[0]
                    lower_value = lower_value1+lower_value2
                nutrient_intake_limit.append(float(upper_value))
                nutrient_must_intake.append(lower_value)
    ocr_df = pd.read_csv(ocr_df_path, encoding='cp949')
    ocr_df.replace(np.nan, 0, inplace=True)

    wanted_ocr_df = ocr_df.loc[:, wanted_column]
    # ✅ 전처리 (1): 단위 전처리
    # 특이 단위를 제외하고 모두 'μg'로 단위 변경

    for col_name in wanted_column1_for_prepro1:
        for row_idx in range(wanted_ocr_df.shape[0]):
            value = wanted_ocr_df[col_name][row_idx]
            if value == 0 :
                continue
            # 숫자 부분 추출
            number_part = float(re.findall(r'[\d\.]+', value)[0])
            # 문자 부분 추출
            text_part = re.findall(r'[^\d\.]+', value)[0].strip().replace(" ", "")
            # 모두 μg으로 변경하기
            if text_part == 'μg':
                changed_value = number_part
            if text_part == 'mg':
                changed_value = number_part * 1000
            if text_part == 'g':
                changed_value = number_part * 1000000
            wanted_ocr_df.loc[row_idx, col_name] = changed_value

    for col_name in ['비타민A', '레티놀팔미테이트(비타민A)', '레티놀아세테이트(비타민A)']:
        for row_idx in range(wanted_ocr_df.shape[0]):
            value = wanted_ocr_df.loc[row_idx, col_name]
            if value == 0 :
                continue
            # 숫자 부분 추출
            number_part = float(re.findall(r'[\d\.]+', value)[0])
            # 문자 부분 추출
            text_part = re.findall(r'[^\d\.]+', value)[0].strip().replace(" ", "")
            wanted_ocr_df.loc[row_idx, col_name] =  number_part

    for row_idx in range(wanted_ocr_df.shape[0]):
        value = wanted_ocr_df['베타카로틴(비타민A)'][row_idx]
        if value == 0 :
            continue
        # 숫자 부분 추출
        number_part = float(re.findall(r'[\d\.]+', value)[0])
        # 문자 부분 추출
        text_part = re.findall(r'[^\d\.]+', value)[0].strip().replace(" ", "")
        # 우선 g과 mg을 μg으로 변경
        if text_part == 'μg':
            changed_value = number_part
        if text_part == 'mg':
            changed_value = number_part * 1000
        if text_part == 'g':
            changed_value = number_part * 1000000
        # μg을 μgRAE으로 변경 (베타카로틴의 경우, 1 μgRAE는 약 0.5 μg의 베타카로틴에 해당)
        changed_value = changed_value * 2
        wanted_ocr_df['베타카로틴(비타민A)'][row_idx] = changed_value

    for col_name in ['비타민E', '비타민E(디알파토코페릴숙시네이트)', '비타민E(디엘알파토코페릴아세테이트)', '비타민E(디알파토코페릴아세테이트)', '비타민E(디알파코페릴숙시네이트)', '비타민E(디알파토코페롤)', '비타민E(디알팤페릴아세테이트)', '비타민E(디엘알파토코페릴숙시네이트)']:
        for row_idx in range(wanted_ocr_df.shape[0]):
            value = wanted_ocr_df.loc[row_idx, col_name]
            if value == 0 :
                continue
            # 숫자 부분 추출
            number_part = float(re.findall(r'[\d\.]+', value)[0])
            # 문자 부분 추출
            text_part = re.findall(r'[^\d\.]+', value)[0].strip().replace(" ", "")
            wanted_ocr_df.loc[row_idx, col_name] = number_part

    for col_name in ['나이아신(비타민B3)', '비타민B3(나이아신아마이드)', '비타민B3(나이아신)', '비타민B3(나이아신마이드)']:
        for row_idx in range(wanted_ocr_df.shape[0]):
            value = wanted_ocr_df.loc[row_idx, col_name]
            if value == 0 :
                continue
            # 숫자 부분 추출
            number_part = float(re.findall(r'[\d\.]+', value)[0])
            # 문자 부분 추출
            text_part = re.findall(r'[^\d\.]+', value)[0].strip().replace(" ", "")
            wanted_ocr_df.loc[row_idx, col_name] = number_part

    # ✅ 전처리 (2): 동일한 성분은 모두 더해주기
    ocr_preprocessed_df = wanted_ocr_df.copy()

    ocr_preprocessed_df['탄수화물 (μg)'] = wanted_ocr_df.iloc[:, 0]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,1:] # 위에 우변의 개수임
    ocr_preprocessed_df['식이섬유 (μg)'] = wanted_ocr_df.iloc[:, 1]+wanted_ocr_df.iloc[:, 2]+wanted_ocr_df.iloc[:, 3]+wanted_ocr_df.iloc[:, 4]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,4:]
    ocr_preprocessed_df['오메가3(EPA+DHA) (μg)'] = wanted_ocr_df.iloc[:, 5]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,1:]
    ocr_preprocessed_df['단백질 (μg)'] = wanted_ocr_df.iloc[:, 6]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,1:]
    ocr_preprocessed_df['메티오닌+시스테인 (μg)'] = wanted_ocr_df.iloc[:, 7]+wanted_ocr_df.iloc[:, 8]+wanted_ocr_df.iloc[:, 9]+wanted_ocr_df.iloc[:, 10]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,4:]
    ocr_preprocessed_df['비타민A (μg RAE)'] = wanted_ocr_df.iloc[:, 11]+wanted_ocr_df.iloc[:, 12]+wanted_ocr_df.iloc[:, 13]+wanted_ocr_df.iloc[:, 14]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,4:]
    ocr_preprocessed_df['비타민D (μg)'] = wanted_ocr_df.iloc[:, 15]+wanted_ocr_df.iloc[:, 16]+wanted_ocr_df.iloc[:, 17]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,3:]
    ocr_preprocessed_df['비타민E (mg α-TE)'] = wanted_ocr_df.iloc[:, 18]+wanted_ocr_df.iloc[:, 19]+wanted_ocr_df.iloc[:, 20]+wanted_ocr_df.iloc[:, 21]+wanted_ocr_df.iloc[:, 22]+wanted_ocr_df.iloc[:, 23]+wanted_ocr_df.iloc[:, 24]+wanted_ocr_df.iloc[:, 25]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,8:]
    ocr_preprocessed_df['비타민K (μg)'] = wanted_ocr_df.iloc[:, 26]+wanted_ocr_df.iloc[:, 27]+wanted_ocr_df.iloc[:, 28]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,3:]
    ocr_preprocessed_df['비타민C (μg)'] = wanted_ocr_df.iloc[:, 29]+wanted_ocr_df.iloc[:, 30]+wanted_ocr_df.iloc[:, 31]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,3:]
    ocr_preprocessed_df['비타민B1 (μg)'] = wanted_ocr_df.iloc[:, 32]+wanted_ocr_df.iloc[:, 33]+wanted_ocr_df.iloc[:, 34]+wanted_ocr_df.iloc[:, 35]+wanted_ocr_df.iloc[:, 36]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,5:]
    ocr_preprocessed_df['비타민B2 (μg)'] = wanted_ocr_df.iloc[:, 37]+wanted_ocr_df.iloc[:, 38]+wanted_ocr_df.iloc[:, 39]+wanted_ocr_df.iloc[:, 40]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,4:]
    ocr_preprocessed_df['비타민B3 (mg NE)'] = wanted_ocr_df.iloc[:, 41]+wanted_ocr_df.iloc[:, 42]+wanted_ocr_df.iloc[:, 43]+wanted_ocr_df.iloc[:, 44]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,4:]
    ocr_preprocessed_df['비타민B5 (μg)'] = wanted_ocr_df.iloc[:, 45]+wanted_ocr_df.iloc[:, 46]+wanted_ocr_df.iloc[:, 47]+wanted_ocr_df.iloc[:, 48]+wanted_ocr_df.iloc[:, 49]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,5:]
    ocr_preprocessed_df['비타민B6 (μg)'] = wanted_ocr_df.iloc[:, 50]+wanted_ocr_df.iloc[:, 51]+wanted_ocr_df.iloc[:, 52]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,3:]
    ocr_preprocessed_df['비타민B7 (μg)'] = wanted_ocr_df.iloc[:, 53]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,1:]
    ocr_preprocessed_df['비타민B9 (μg)'] = wanted_ocr_df.iloc[:, 54]+wanted_ocr_df.iloc[:, 55]+wanted_ocr_df.iloc[:, 56]+wanted_ocr_df.iloc[:, 57]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,4:]
    ocr_preprocessed_df['비타민B12 (μg)'] = wanted_ocr_df.iloc[:, 58]+wanted_ocr_df.iloc[:, 59]+wanted_ocr_df.iloc[:, 60]+wanted_ocr_df.iloc[:, 61]+wanted_ocr_df.iloc[:, 62]+wanted_ocr_df.iloc[:, 63]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,6:]
    ocr_preprocessed_df['칼슘 (μg)'] = wanted_ocr_df.iloc[:, 64]+wanted_ocr_df.iloc[:, 65]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,2:]
    ocr_preprocessed_df['인 (μg)'] = wanted_ocr_df.iloc[:, 66]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,1:]
    ocr_preprocessed_df['나트륨 (μg)'] = wanted_ocr_df.iloc[:, 67]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,1:]
    ocr_preprocessed_df['염소 (μg)'] = wanted_ocr_df.iloc[:, 68]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,1:]
    ocr_preprocessed_df['칼륨 (μg)'] = wanted_ocr_df.iloc[:, 69]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,1:]
    ocr_preprocessed_df['마그네슘 (μg)'] = wanted_ocr_df.iloc[:, 70]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,1:]
    ocr_preprocessed_df['철 (μg)'] = wanted_ocr_df.iloc[:, 71]+wanted_ocr_df.iloc[:, 72]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,2:]
    ocr_preprocessed_df['아연 (μg)'] = wanted_ocr_df.iloc[:, 73]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,1:]
    ocr_preprocessed_df['구리 (μg)'] = wanted_ocr_df.iloc[:, 74]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,1:]
    ocr_preprocessed_df['망간 (μg)'] = wanted_ocr_df.iloc[:, 75]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,1:]
    ocr_preprocessed_df['요오드 (μg)'] = wanted_ocr_df.iloc[:, 76]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,1:]
    ocr_preprocessed_df['셀레늄 (μg)'] = wanted_ocr_df.iloc[:, 77]+wanted_ocr_df.iloc[:, 78]+wanted_ocr_df.iloc[:, 79]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,3:]
    ocr_preprocessed_df['몰리브덴 (μg)'] = wanted_ocr_df.iloc[:, 80]+wanted_ocr_df.iloc[:, 81]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,2:]
    ocr_preprocessed_df['크롬 (μg)'] = wanted_ocr_df.iloc[:, 82]+wanted_ocr_df.iloc[:, 83]+wanted_ocr_df.iloc[:, 84]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,3:]
    ocr_preprocessed_df['루테인 (μg)'] = wanted_ocr_df.iloc[:, 85]+wanted_ocr_df.iloc[:, 86]+wanted_ocr_df.iloc[:, 87]+wanted_ocr_df.iloc[:, 88]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,4:]
    ocr_preprocessed_df['지아잔틴 (μg)'] = wanted_ocr_df.iloc[:, 89]
    ocr_preprocessed_df = ocr_preprocessed_df.iloc[:,1:]

    nutrition_being_taken = [0 for _ in range(ocr_preprocessed_df.shape[1])]

    for row in range(ocr_preprocessed_df.shape[0]):
        row_list = ocr_preprocessed_df.iloc[row].tolist()
        nutrition_being_taken = [x + y for x, y in zip(nutrition_being_taken, row_list)]

    # 상한/하한 조정
    nutrient_intake_limit = [x - y for x, y in zip(nutrient_intake_limit, nutrition_being_taken)]
    nutrient_must_intake = [x - y for x, y in zip(nutrient_must_intake, nutrition_being_taken)]
    component_list = preprocessed_df.values.tolist()

    # # 위 데이터를 숫자로 변환
    # # dict에서 value 값을 preprocessed_df의 칼럼 인덱스 번호로 바꿔주기
    # preprocessed_df_columns_list = preprocessed_df.columns.tolist()
    # preprocessed_df_columns_list = [re.sub(r' \(.*?\)', '', item) for item in preprocessed_df_columns_list]
    # health_concern_with_ingredient_data_as_list = []

    # for k, v in health_concern_with_ingredient_data.items():
    #     col_idx_list = []
    #     for value in v:
    #         col_idx = preprocessed_df_columns_list.index(value)
    #         col_idx_list.append(col_idx)
    #     health_concern_with_ingredient_data[k]=col_idx_list


    nutrient_must_intake_idx = []
    for preference_idx in user_1['preference_category']:
        preference_name = list(health_concern_with_ingredient_data)[preference_idx]
        nutrient_must_intake_idx.extend(health_concern_with_ingredient_data[preference_name])
    nutrient_must_intake = [nutrient_must_intake[i] if i in nutrient_must_intake_idx else 0 for i in range(len(nutrient_must_intake))]


    # 🌷 Generate population
    def generate_population(nutrient_num, user_info_dict, population_size):
        '''
        한 세대 생성 (중첩 리스트)
        ✔ nutrient_num: [int] 우리가 가진 전체 영양제 제품 수 (해당 수가 한 chromosome 크기)
        ✔ user_info_dict: [dict] 사용자 정보를 dictionary에 저장
        ✔ population_size: [int] 한 세대에 있는 chromosome 수

        ◽ 전체 10개를 선택
        ◽ 10/N개 만큼 각 카테고리에서 선택
        ◽ 나머지는 전체 중 선택
        '''
        population = []
        for _ in range(population_size):
            # ✔ chromosome 초기화
            chromosome = [0 for _ in range(nutrient_num)]
            user_preference_category_list = user_info_dict['preference_category']
            num_per_category = 10//len(user_preference_category_list) # 사용자가 선택한 카테고리에서 몇 개 고를지
            num_per_total = 10%len(user_preference_category_list)

            # ✔ 사용자가 선택한 카테고리에서 num_per_category개 만큼 고르기
            for prefer_category in user_preference_category_list:
                # nutrient category에서 값이 prefer_category에 해당하는 인덱스 찾기 (🤗카테고리 수정에 따른 수정)
                index= [idx for idx, sublist in enumerate(nutrient_category) if prefer_category in sublist]

                # ✔ 해당 인덱스에서 무작위로 하나 선정
                random_index = random.sample(index, num_per_category)
                # 해당 영양제는 선택하기
                for idx in random_index:
                    chromosome[idx] = 1
            # ✔ 전체 중 num_per_total개 고르기
            random_index = random.sample(range(0,nutrient_num), num_per_total)
            for idx in random_index:
                chromosome[idx] = 1
            population.append(chromosome)
        return population


    # 🌷 Fitness function
    def evaluate_fitness(chromosome, user_info_dict, standard_1, standard_2):
        '''
        ✔ user_info_dict: [dict] 사용자 정보를 dictionary에 저장
        ✔ standard_1: [list] 각 영양성분 별로 섭취 한계량이 담긴 리스트 (현재 변수명: nutrient_intake_limit) 🔥 이것도 나중에 임산부/남/녀/연령 등에 따라서 다른 기준으로 입력 받기
        ✔ standard_2: [list] 각 영양성분 별로 평균 필요량이 담긴 리스트 (현재 변수명: nutrient_must_intake)

        ◽ 합이 가장 작은 것을 반환
        ◽ 패널티 = 10000000000000000000
            1. 사용자 선호하는 카테고리가 없을 경우 (crossover, mutation 시 숫자 바뀔 수 있으므로)
            2. 사용자의 성별에 위반되는 경우
            3. 섭취 한계를 넘어선 경우
            4. 평균필요량을 만족하지 못한 경우

            [24.05.31 (금) 피드백] 📌🐥📌🐥📌🐥📌🐥📌🐥📌🐥📌🐥📌🐥📌🐥📌🐥📌🐥📌🐥📌🐥
            ✔ 성분별 섭취량 >= 필요 평균량 & <= 섭취 한계 => 선택된 영양제 수 minimize
            ✔ minimize로 한 다음에 영양제가 너무 다양하지 않으면, 최소 영양제를 몇 개 선택하는지의 조건을 걸어주면 좋을 듯
            ✔ minimize하는 또다른 이유는, '상한섭취량' 제한이 없는 영양성분 때문
            (https://kormedi.com/1414916/%EC%83%81%ED%95%9C%EC%84%AD%EC%B7%A8%EB%9F%89-%EC%97%86%EB%8A%94-%EB%B9%84%ED%83%80%EB%AF%BC-%EB%AC%B4%EC%A1%B0%EA%B1%B4-%EB%A7%8E%EC%9D%B4-%EB%A8%B9%EC%96%B4%EB%8F%84-%EB%90%A0%EA%B9%8C/#google_vignette)
        '''
        fitness_score = 10000000000000000000 # 초기값


        # # ✔ 패널티 (1)
        # user_preference_category_list = user_info_dict['preference_category']
        # for prefer_category in user_preference_category_list:
        #     # nutrient category에서 값이 prefer_category에 해당하는 인덱스 찾기(🤗카테고리 수정에 따른 수정)
        #     index= [idx for idx, sublist in enumerate(nutrient_category) if prefer_category in sublist]
        #     result = [chromosome[i] for i in index]
        #     if 1 not in result:
        #         fitness_score = 10000000000000000000
        #         return fitness_score

        # ✔ 패널티 (2)
        user_gender = user_info_dict['gender']
        if user_gender == 0: # 사용자=여자
            exclusion = [i for i, x in enumerate(nutrient_gender_list) if x == 1]
        else: # 남자
            exclusion = [i for i, x in enumerate(nutrient_gender_list) if x == 0]
        selected_nutrient_index = [i for i, x in enumerate(chromosome) if x == 1] # 크로모좀 값이 1인 것이 선택된 것임
        # 선택된 영양제 인덱스가 반대 성별에 해당하는 것이면 패널티
        common_values = set(selected_nutrient_index) & set(exclusion)
        if common_values:
            fitness_score = 10000000000000000000
            return fitness_score

        # ✔ 패널티 (3) & 패널티 (4)
        # 📌 standard_2(평균 필요량)를 '사용자가 설정한 카테고리'만 채우면 되는걸로 수정 (아래 코드를 위에 사용자 입력란에 추가함)
        # standard_2 = [standard_2[i] if i in user_info_dict['preference_category'] else 0 for i in range(len(standard_2))]
        selected_nutrient = [component_list[i] for i in selected_nutrient_index]
        for component_idx in range(len(component_list[0])):
            sum_of_component = sum(sublist[component_idx] for sublist in selected_nutrient)
            if sum_of_component > standard_1[component_idx]: # 상한
                fitness_score = 10000000000000000000
                return fitness_score
            if sum_of_component < standard_2[component_idx]: # 하한
                fitness_score = 10000000000000000000
                return fitness_score
        

        # # ✔ 패널티 (5)
        selected_nutrient_num = sum(chromosome)
        if selected_nutrient_num <= 1:
            fitness_score = 10000000000000000000
            return fitness_score

        # ✔ penalty가 아닌 경우의 fitness value 값 계산
        # fitness value 계산 방법
        # 📌 5027015.23 📌 이렇게 나오면, 5개의 영양제가 선택됐고, 27개의 성분이 채워져 있고, 각 성분별 %의 합이 15.23이라는 의미 (즉, 만약에 성분 1이 다 채워지면100%이므로 1임)
        # (1) 영양 성분 중에 0이 있는지 확인하기 (즉, 몇 개의 영양소가 채워져 있는지 확인)
            # 지금은 성분 개수가 30개임 (🔥 변수명: nutrient_intake_limit => len(nutrient_intake_limit)로 해야 함)
        # (2) 해당 영양소 별로 몇 % 만족했는지 확인하기
            ### 🔥 몇몇 상한 섭취량은 float('inf')이므로, 이를 나눴을 때 값은 0이 됨

        selected_nutrient_num = sum(chromosome)

        # selected_nutrient: 중첩 리스트 > 내부 리스트: 선택된 각 영양소별 성분이 담겨있음
        accumulate_nutrient_info = [sum(sublist) for sublist in zip(*selected_nutrient)]
        satisfied_component_num = sum(1 for num in accumulate_nutrient_info if num != 0) # (1) 관련 변수
        satisfied_component_ratio = [x/y for x, y in zip(accumulate_nutrient_info, nutrient_intake_limit)] # (2) 관련 변수
        accumulated_satisfied_component_ratio = sum(satisfied_component_ratio)

        fitness_score = selected_nutrient_num*100000 - satisfied_component_num - accumulated_satisfied_component_ratio*0.0001

        # [24.06.22] 📌⭐📌⭐📌⭐📌⭐ 카테고리 다양성을 위해서 (2)는 제거해봄 (별로임 -> 없앰)
        # fitness_score = selected_nutrient_num*100000 - satisfied_component_num
        return fitness_score


    # 🌷 Crossover
    def crossover(chromosome1, chromosome2):
        '''
        ◽ two-point crossover
        ◽ 사용자 성별이 아닌 것은 0으로 바꾸지 않음 (나중에 fitness_value=0으로 처리하면 됨)
        '''
        chromosome_len = len(chromosome1)
        crossover_point1, crossover_point2 = random.sample(range(1, chromosome_len), 2)
        children=[]

        child1 = chromosome1[:crossover_point1+1]
        child1[crossover_point1+1 : crossover_point2+1] = chromosome2[crossover_point1+1 : crossover_point2+1]
        child1[crossover_point2+1:] = chromosome1[crossover_point2+1:]
        children.append(child1)

        child2 = chromosome2[:crossover_point1+1]
        child2[crossover_point1+1 : crossover_point2+1] = chromosome1[crossover_point1+1 : crossover_point2+1]
        child2[crossover_point2+1:] = chromosome2[crossover_point2+1:]
        children.append(child2)

        return children


    # 🌷 Mutation (무조건 돌연변이 한 곳에서 발생)
    # 이거는 성별 아닌거 0으로 바꾸지 말기 (그냥 패널티 받게)
    # 🔥 나중에 다른 방법으로 바꿔보기
    def mutation(child):
        mutation_point = random.choice(range(len(child)))
        if child[mutation_point] == 0:
            child[mutation_point] = 1
        else:
            child[mutation_point] = 0
        return child

    # random.seed(42) # 🐾
    # np.random.seed(42) # 📌

    nutrient_name_list = nutrient_data['브랜드명_제품명'].tolist()
    nutrient_num = preprocessed_df.shape[0]

    ############################################################################
    # 🌷 Initialize population
    initial_population = generate_population(nutrient_num, user_1, initial_population_num)

    # 🌷 Fitness function value calculation & Selection & Crossover & Mutation
    # 세대 수 (5000 ~ 10000)
    generation_num = generation_num # 🔥 우선 시험삼아 100세대만 진행

    # 초기 세대
    current_population = initial_population[:]
    # 최대 적합도 초기화
    best_score = 10000000000000000000

    no_improvement_limit = 100  # 개선이 없을 때 종료할 세대 수
    no_improvement_counter = 0  # 개선이 없는 세대 수를 세는 변수

    # 🌷 GA 시작
    for _ in range(generation_num):
        # ✅ 적합도 평가 수행
        fitness_result = [evaluate_fitness(chromosome, user_1, nutrient_intake_limit, nutrient_must_intake) for chromosome in current_population]

        # ✔ 최대적합도 업데이트
        if min(fitness_result) < best_score:
            best_score = min(fitness_result)
            best_score_idx = fitness_result.index(best_score)
            best_chromosome = current_population[best_score_idx]
            no_improvement_counter = 0  # 적합도 점수가 개선되었으므로 초기화
        else:
            no_improvement_counter += 1  # 개선되지 않으면 증가

        if no_improvement_counter >= no_improvement_limit:
            print(f"조기 종료 조건 만족: {_}세대에서 종료")
            break

        # ✅ <Selection>
        # fitness_result 오름차순 정렬
        sorted_indices = sorted(range(len(fitness_result)), key=lambda i: fitness_result[i])
        # 정렬된 fitness_result 값과 매칭되는 current_population 구하기
        sorted_chromosome = [current_population[i] for i in sorted_indices]
        # 상위 20%는 다음 세대로
        new_population = sorted_chromosome[:200]

        # ✅ <Crossover>
        # 30%만 crossover 진행 (300개는 crossover, 700개는 selection)
            # 1-2 크로모좀 크로스오버
            # 3-4 크로모좀 크로스오버
        crossover_population = []
        crossover_and_selection_population = []
        for idx in range(0, 300, 2):
            parent_1 = current_population[idx]
            parent_2 = current_population[idx+1]
            children = crossover(parent_1, parent_2)
            crossover_population.extend(children)
        # crossover에 참여하지 않은 나머지 chromosome은 그냥 추가하기
        crossover_and_selection_population = crossover_population[:]
        non_crossover_chromosome = current_population[300:]
        crossover_and_selection_population.extend(non_crossover_chromosome)

        # ✔ Crossover 대상 적합도 검사 수행
        crossover_fitness_result = [evaluate_fitness(chromosome, user_1, nutrient_intake_limit, nutrient_must_intake) for chromosome in crossover_and_selection_population]

        # 상위 20%는 다음 세대로
        sorted_indices = sorted(range(len(crossover_fitness_result)), key=lambda i: crossover_fitness_result[i])
        sorted_chromosome = [crossover_and_selection_population[i] for i in sorted_indices]
        top_200 = sorted_chromosome[:200]
        new_population.extend(top_200)

        # ✅ <Mutation>
        mutation_population = copy.deepcopy(crossover_population)
        # crossover만 된 애들 중에서 90개 mutation 진행
        mutation_idx_list = np.random.choice(300, 90, replace = False)
        mutation_idx_list = list(mutation_idx_list)
        for idx in mutation_idx_list:
            mutation_child = mutation(mutation_population[idx])
            mutation_population[idx] = mutation_child
        # selection만 된 700개 추가
        mutation_population.extend(non_crossover_chromosome)

        # ✔ Mutation 대상 적합도 검사 수행
        mutation_fitness_result = [evaluate_fitness(chromosome, user_1, nutrient_intake_limit, nutrient_must_intake) for chromosome in mutation_population]

        # 상위 10%는 다음 세대로
        sorted_indices = sorted(range(len(mutation_fitness_result)), key=lambda i: mutation_fitness_result[i])
        sorted_chromosome = [mutation_population[i] for i in sorted_indices]
        top_100 = sorted_chromosome[:100]
        new_population.extend(top_100)

        # ✅ 다음 세대 생성 (50%는 랜덤으로 생성)
        next_population = generate_population(nutrient_num, user_1, 500)
        new_population.extend(next_population)

        current_population = new_population

    if best_score == 10000000000000000000:
        best_chromosome = None
    return best_score, best_chromosome, nutrient_name_list








