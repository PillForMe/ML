# 필요한 라이브러리 임포트
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, LlamaForSequenceClassification
import pandas as pd
import random
import numpy as np
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import itertools

app = Flask(__name__)
CORS(app)  # 모든 도메인에서의 요청을 허용


item_info = pd.read_excel('item_info.xlsx')

model = None
tokenizer = None

def get_model():
    try:
        g.model
        g.tokenizer
    except:
        g.model = AutoModelForSequenceClassification.from_pretrained("./test")
        # Move model to bfloat16 precision
        g.model = g.model.to(torch.bfloat16)
        # Move model to GPU(s)
        g.model.cuda()
        g.tokenizer = AutoTokenizer.from_pretrained("./test")


    return g.model, g.tokenizer


@app.route('/glrec', methods=['POST'])
def glrec_result():
    model, tokenizer = get_model()

    incoming_data = request.json  # 백엔드로부터 받은 데이터
    print('Received from BE:', incoming_data)

    ga_output = incoming_data['ga_output']
    efficacy = incoming_data['efficacy']

    # 2개씩 묶은 조합 만들기
    items = []
    for comb in ga_output:
        for i in range(len(comb)):
            items.append(comb[i])
    items = list(set(items))

    result_comb  = list(itertools.combinations(items, 2))

    # 각 조합에서 가능한 모든 리스트 생성
    # result_comb = []
    # for comb in combinations:
    #     list1, list2 = comb
    #     product_combinations = list(itertools.product(list1, list2))
    #     result_comb.extend(product_combinations)

    basic_prompt = """
    ### 질문: 7606번의 유저에게 이 A를 추천할까요? 아니면 B를 추천할까요?

    ### 맥락:7606번의 유저는 성별:여\/연령대:20대\/특이사항:없음\/관심 있는 건강고민 정보:눈건강,장건강,스트레스&수면\/관심 있는 영양제 효능 정보:{efficacy}의 특성을 가지고 있다.
    7606번의 유저는'안국건강_안국 루테인 지아잔틴 미니'라는 이름의 영양제를 구매한 이력이 있다.
    이 영양제의 효능:눈 피로감 개선,안구건조증 개선,야맹증 개선/이 영양제와 관련 있는 건강고민 정보:눈건강_황반색소유지
    이 영양제를 구매한 또 다른 사람은 성별:남\/연령대:10대\/특이사항:없음\/관심 있는 건강고민 정보:눈건강\/관심 있는 영양제 효능 정보:눈 피로감 개선,시력 개선,야맹증 개선,눈 가려움 완화,안구건조증 개선의 특성을 지난다.
    7606번의 유저는'암웨이 뉴트리라이트_밸런스 위드인 365 프로바이오틱스'라는 이름의 영양제를 구매한 이력이 있다.
    이 영양제의 효능:변비 개선,복부 가스 덜 참,설사 빈도 감소/이 영양제와 관련 있는 건강고민 정보:장건강_유익균유해균균형도움
    이 영양제를 구매한 또 다른 사람은 성별:남\/연령대:20대\/특이사항:없음\/관심 있는 건강고민 정보:피부건강,장건강\/관심 있는 영양제 효능 정보:아랫배 통증 완화,설사 빈도 감소,뾰루지 감소의 특성을 지난다.
    7606번의 유저가 구매를 고민하고 있는 영양제 A와 B의 정보는 다음과 같다:
    """.strip()

    sentences= []
    for comb in result_comb:
        a = comb[0]
        b = comb[1]
        try:
            a_item_info_value = item_info.loc[item_info['브랜드명_제품명'] == a, 'item_info'].values[0]
            b_item_info_value = item_info.loc[item_info['브랜드명_제품명'] == b, 'item_info'].values[0]

            compare = f"""
            'B'라는 이름의 영양제는 다음과 같은 특성을 가진다.
            {b_item_info_value}
            'A'라는 이름의 영양제는 다음과 같은 특성을 가진다.
            {a_item_info_value}
            ### 답변:
            """
            sentences.append((a, b,basic_prompt + compare))
        except:
            pass

    glrec_dict = {}
    for item in items:
        glrec_dict[item] = 0

    for a,b,sentence in sentences:
        # 배치 인퍼런스를 위한 입력 텍스트
        input_texts = sentence

        # 토크나이저로 입력 텍스트들을 배치로 변환
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)

        # GPU로 이동
        inputs = {key: val.cuda() for key, val in inputs.items()}

        # 모델 예측
        with torch.inference_mode():
            outputs = model(**inputs, output_attentions=True)

        # 결과값(로짓)을 소프트맥스 함수에 넣어서 확률로 변환
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        print(probabilities)

        # 가장 높은 확률을 가진 라벨 선택
        predicted_labels = torch.argmax(probabilities, dim=-1).cpu().numpy()
        # A가 1, B가 0

        print(f"Predicted labels: {predicted_labels}")

        if predicted_labels[0] == 1:
            if a in glrec_dict:
                glrec_dict[a] += 1
            else:
                glrec_dict[a] = 1
        else:
            if b in glrec_dict:
                glrec_dict[b] += 1
            else:
                glrec_dict[b] = 1
        
        # 텐서 해제 및 GPU 메모리 정리
        del inputs
        del outputs
        del logits
        del probabilities
        torch.cuda.empty_cache()

    del model
    del tokenizer

    # glrec_result = {}
    # for comb in ga_output:
    #     item_a = comb[0]
    #     item_b = comb[1]
    #     score = 0
    #     score += glrec_dict[item_a]
    #     score += glrec_dict[item_b]
    #     glrec_result[comb] = score

    result = {"glrec_result": glrec_dict}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7000)