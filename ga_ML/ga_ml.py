import pandas as pd
import numpy as np
import random
from multiprocessing import Pool, Manager
from ga_func import ga
from flask import Flask, request, jsonify
import socket
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 모든 도메인에서의 요청을 허용

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
ocr_df_path = "./ocr_result1.csv" ## 👨🏻‍💻 경로 변경

random.seed(42) #
np.random.seed(42) #

def run_ga(process_id, user, nutrient_raw_data, item_health_grouped_df_path, limitation_df_path, ocr_df_path, generation_num, initial_population_num, return_dict):
    print(f"Process {process_id} started.")
    # nutrient_raw_data의 65%를 랜덤 샘플링
    sampled_data = nutrient_raw_data.sample(frac=0.8, random_state=process_id)
    ga_best_score, ga_best_chromosome, ga_nutrient_name_list = ga(user, sampled_data, item_health_grouped_df_path, limitation_df_path, ocr_df_path, generation_num, initial_population_num)

    # 결과 저장
    if ga_best_score != 10000000000000000000:
        best_chromosome_indices = [index for index, value in enumerate(ga_best_chromosome) if value == 1]
        nutrient_name = [value for index, value in enumerate(ga_nutrient_name_list) if index in best_chromosome_indices]
        return_dict[process_id] = (nutrient_name, ga_best_score)
        print(f"Process {process_id} completed successfully.")
    else:
        return_dict[process_id] = (None, None)
        print(f"Process {process_id} failed.")

# user = {
#     'gender': 0,
#     'age': 29,
#     'condition': 0,
#     'preference_category' : [2, 7, 8]
# }

@app.route('/ga', methods=['POST'])
def GA():
    print("request recieved")
    incoming_data = request.json  # 백엔드로부터 받은 데이터
    print('Received from BE:', incoming_data)

    user_data = incoming_data['user']
    user = user_data

    if __name__ == "__main__":
        num_processes = 15
        generation_num = 3500
        initial_population_num = 1000

        # Manager를 통해 프로세스 간의 결과를 공유할 수 있는 dict 생성
        manager = Manager()
        return_dict = manager.dict()

        # Pool을 사용하여 병렬 처리
        pool = Pool(processes=num_processes)

        processes = []
        for i in range(num_processes):
            process = pool.apply_async(run_ga, args=(i, user, nutrient_raw_data, item_health_grouped_df_path, limitation_df_path, ocr_df_path, generation_num, initial_population_num, return_dict))
            processes.append(process)

        # 모든 프로세스가 끝날 때까지 대기
        for process in processes:
            process.get()

        pool.close()
        pool.join()

        ga_result = []
        ga_score_collection = []

        # 결과 출력 및 정리
        for i in range(num_processes):
            nutrient_name, ga_best_score = return_dict[i]
            if nutrient_name is not None:
                print(f"📌 {i+1}번 조합: {nutrient_name}")
                ga_result.append(nutrient_name)
                ga_score_collection.append(ga_best_score)
            else:
                print(f"💥 {i+1}번 조합 실패")

    # 결과 출력 및 정리
    unique_ga_result = []
    unique_ga_score_collection = []

    # 튜플로 변환하여 중복 제거
    seen = set()
    for i in range(len(ga_result)):
        t = tuple(sorted(ga_result[i]))  # 리스트를 튜플로 변환
        if t not in seen:
            seen.add(t)
            unique_ga_result.append(ga_result[i])
            unique_ga_score_collection.append(ga_score_collection[i])

    print(unique_ga_score_collection)
    print(len(unique_ga_result))

    result = {"ga_output": unique_ga_result}

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6000)