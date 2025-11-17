# NBA_prediction
1. 파일 구성

    1.1. /data
    - 연구에 사용한 데이터셋 원본
    
    1.2. extract_emoji_score.py
    - 크롤링으로 수집한 커뮤니티 게시글 데이터에 시간 구간 매핑
    - 감성 사전 감성점수, ELECTRA 감성점수를 추출
        
        1.2.1. extract_model.py
        - KcELECTRA 모델 감성점수 추출
        
        1.2.2. extract_emojidic.py
        - KNU 감성사전 감성점수 추출
        
    1.3. extract_sentiment_features.py
    - 구간별 감성점수, 부상 여부, 팀별 감성격차를 추출

    1.4. compare_models.py
    - 모든 독립변수가 추가된 데이터로 XGBoost, Random Forest, LSTM 
      세 모델별 성능 비교
