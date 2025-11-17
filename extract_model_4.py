# -*- coding: utf-8 -*-

import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm
from huggingface_hub import login

class ELECTRASentimentAnalyzer:
    """
    한국어 ELECTRA 모델을 이용한 감성 분석 클래스
    """

    def __init__(self, model_nm='beomi/KcELECTRA-base-v2022'):
        """
        ELECTRA 모델 초기화
        기본값: KcELECTRA (한국어 감성분석에 최적화된 모델)
        다른 옵션:
        - 'beomi/kcbert-base'
        - 'monologg/kobert'
        - 'klue/roberta-base'
        """
        print(f"BERT 모델 로딩 중: {model_nm}")


        login("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")

        model_nm = 'beomi/KcELECTRA-base-v2022'
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"사용 디바이스: {self.device}")

            self.tokenizer = AutoTokenizer.from_pretrained(model_nm, trust_remote_code=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_nm  # 감성분석용 파인튜닝 모델
            )
            self.model.to(self.device)
            self.model.eval()

            print("모델 로드 완료!")
        except Exception as e:
            print(f"모델 로드 중 오류: {e}")
            print("기본 모델로 재시도...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_nm)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_nm,
                    num_labels=2
                )
                self.model.to(self.device)
                self.model.eval()
                print("기본 모델 로드 완료!")
            except Exception as e2:
                print(f"기본 모델 로드 실패: {e2}")
                self.model = None
                self.tokenizer = None

    def analyze_sentiment(self, text, max_length=512):
        """
        텍스트의 감성을 분석하여 점수 반환
        Returns: {
            'positive_prob': 긍정 확률,
            'negative_prob': 부정 확률,
            'sentiment_score': 감성 점수 (-1 ~ 1),
            'label': 'positive' or 'negative'
        }
        """
        if not self.model or not self.tokenizer:
            return {
                'positive_prob': 0.0,
                'negative_prob': 0.0,
                'sentiment_score': 0.0,
                'label': 'neutral'
            }

        try:
            # 텍스트가 너무 길면 앞부분만 사용
            if len(text) > 1000:
                text = text[:1000]

            # 토큰화
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding=True
            )
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            # 예측
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            # 결과 해석 (0: 부정, 1: 긍정)
            negative_prob = float(probs[0])
            positive_prob = float(probs[1])

            # 감성 점수 계산 (-1 ~ 1 범위)
            sentiment_score = positive_prob - negative_prob

            label = 'positive' if positive_prob > negative_prob else 'negative'

            return {
                'positive_prob': round(positive_prob, 4),
                'negative_prob': round(negative_prob, 4),
                'sentiment_score': round(sentiment_score, 4),
                'label': label
            }

        except Exception as e:
            print(f"감성 분석 중 오류: {e}")
            return {
                'positive_prob': 0.0,
                'negative_prob': 0.0,
                'sentiment_score': 0.0,
                'label': 'error'
            }


def process_json_with_model_sentiment(input_file, output_file):
    """
    JSON 파일을 읽어서 ELECTRA로 감성점수를 추가
    """
    # ELECTRA 감성 분석기 초기화
    analyzer = ELECTRASentimentAnalyzer()

    if not analyzer.model:
        print("ELECTRA 모델을 로드할 수 없습니다.")
        return

    try:
        processed_count = 0
        total_lines = 0

        # 전체 라인 수 카운트
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f if line.strip())

        print(f"총 {total_lines}개 라인 처리 시작...")

        with open(input_file, 'r', encoding='utf-8') as infile, \
                open(output_file, 'w', encoding='utf-8') as outfile:

            # 진행 상황 표시
            for line in tqdm(infile, total=total_lines, desc="감성 분석 진행"):
                line = line.strip()
                if not line:
                    continue

                try:
                    # JSON 파싱
                    data = json.loads(line)

                    # subject와 body 결합
                    text = f"{data.get('subject', '')} {data.get('body', '')}"

                    # ELECTRA로 감성점수 계산
                    sentiment_result = analyzer.analyze_sentiment(text)

                    # 감성점수를 JSON에 추가
                    data['model_sentiment'] = sentiment_result

                    # 결과를 파일에 쓰기
                    json_line = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
                    outfile.write(json_line + '\n')

                    processed_count += 1

                except json.JSONDecodeError as e:
                    print(f"\nJSON 파싱 오류: {e}")
                    continue
                except Exception as e:
                    print(f"\n처리 중 오류 발생: {e}")
                    continue

        print(f"\n처리 완료!")
        print(f"입력 파일: {input_file}")
        print(f"출력 파일: {output_file}")
        print(f"총 처리된 라인: {processed_count}/{total_lines}")

    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {input_file}")
    except Exception as e:
        print(f"파일 처리 중 오류 발생: {e}")


def calculate_team_sentiment_summary(input_file, output_summary_file):
    """
    팀별 감성점수 요약 통계 계산
    """
    team_sentiments = {}

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    team_id = data.get('teams_id', '')

                    if team_id and 'model_sentiment' in data:
                        if team_id not in team_sentiments:
                            team_sentiments[team_id] = []

                        sentiment_score = data['model_sentiment']['sentiment_score']
                        team_sentiments[team_id].append(sentiment_score)

                except Exception as e:
                    continue

        # 팀별 통계 계산
        team_summary = {}
        for team_id, scores in team_sentiments.items():
            team_summary[team_id] = {
                'count': len(scores),
                'avg_sentiment': round(np.mean(scores), 4),
                'std_sentiment': round(np.std(scores), 4),
                'min_sentiment': round(min(scores), 4),
                'max_sentiment': round(max(scores), 4),
                'positive_count': sum(1 for s in scores if s > 0),
                'negative_count': sum(1 for s in scores if s < 0),
                'neutral_count': sum(1 for s in scores if s == 0)
            }

        # 요약 파일 저장
        with open(output_summary_file, 'w', encoding='utf-8') as f:
            json.dump(team_summary, f, ensure_ascii=False, indent=2)

        print(f"\n팀별 감성 요약 저장: {output_summary_file}")
        print("\n팀별 평균 감성점수:")
        for team_id, summary in sorted(team_summary.items(), key=lambda x: x[1]['avg_sentiment'], reverse=True):
            print(f"  {team_id}: {summary['avg_sentiment']:.4f} (총 {summary['count']}개)")

    except Exception as e:
        print(f"팀별 요약 계산 중 오류: {e}")


def predict_match_winner(sentiment_file, team1, team2, output_file='match_prediction.json'):
    """
    두 팀의 감성격차를 분석하여 승패 예측
    """
    team1_scores = []
    team2_scores = []

    team1_details = []
    team2_details = []

    try:
        with open(sentiment_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    team_id = data.get('teams_id', '').lower()

                    if 'model_sentiment' in data:
                        sentiment_score = data['model_sentiment']['sentiment_score']

                        if team_id == team1.lower():
                            team1_scores.append(sentiment_score)
                            team1_details.append({
                                'id': data.get('id', ''),
                                'subject': data.get('subject', ''),
                                'sentiment_score': sentiment_score,
                                'label': data['model_sentiment']['label']
                            })
                        elif team_id == team2.lower():
                            team2_scores.append(sentiment_score)
                            team2_details.append({
                                'id': data.get('id', ''),
                                'subject': data.get('subject', ''),
                                'sentiment_score': sentiment_score,
                                'label': data['model_sentiment']['label']
                            })

                except Exception as e:
                    continue

        if not team1_scores or not team2_scores:
            print(f"\n경고: {team1} 또는 {team2} 팀의 데이터가 충분하지 않습니다.")
            print(f"{team1} 데이터: {len(team1_scores)}개")
            print(f"{team2} 데이터: {len(team2_scores)}개")
            return None

        # 팀별 통계 계산
        team1_avg = np.mean(team1_scores)
        team2_avg = np.mean(team2_scores)

        team1_std = np.std(team1_scores)
        team2_std = np.std(team2_scores)

        team1_positive_rate = sum(1 for s in team1_scores if s > 0) / len(team1_scores)
        team2_positive_rate = sum(1 for s in team2_scores if s > 0) / len(team2_scores)

        # 감성격차 계산
        sentiment_gap = team1_avg - team2_avg

        # 승리 예측
        if sentiment_gap > 0:
            predicted_winner = team1
            win_probability = min(0.5 + abs(sentiment_gap) * 0.5, 0.95)
        else:
            predicted_winner = team2
            win_probability = min(0.5 + abs(sentiment_gap) * 0.5, 0.95)

        # 신뢰도 계산 (데이터 개수가 많을수록, 격차가 클수록 높음)
        min_data_count = min(len(team1_scores), len(team2_scores))
        confidence = min(min_data_count / 100 * abs(sentiment_gap), 1.0)
        confidence_label = 'HIGH' if confidence > 0.7 else 'MEDIUM' if confidence > 0.4 else 'LOW'

        # 결과 정리
        prediction_result = {
            'match_info': {
                'team1': team1.upper(),
                'team2': team2.upper(),
                'analysis_date': str(np.datetime64('now'))
            },
            'sentiment_analysis': {
                team1.upper(): {
                    'data_count': len(team1_scores),
                    'avg_sentiment': round(team1_avg, 4),
                    'std_sentiment': round(team1_std, 4),
                    'positive_rate': round(team1_positive_rate, 4),
                    'min_sentiment': round(min(team1_scores), 4),
                    'max_sentiment': round(max(team1_scores), 4)
                },
                team2.upper(): {
                    'data_count': len(team2_scores),
                    'avg_sentiment': round(team2_avg, 4),
                    'std_sentiment': round(team2_std, 4),
                    'positive_rate': round(team2_positive_rate, 4),
                    'min_sentiment': round(min(team2_scores), 4),
                    'max_sentiment': round(max(team2_scores), 4)
                }
            },
            'prediction': {
                'sentiment_gap': round(sentiment_gap, 4),
                'predicted_winner': predicted_winner.upper(),
                'win_probability': round(win_probability, 4),
                'confidence_level': round(confidence, 4),
                'confidence_label': 'HIGH' if confidence > 0.7 else 'MEDIUM' if confidence > 0.4 else 'LOW'
            },
            'top_positive_comments': {
                team1.upper(): sorted(team1_details, key=lambda x: x['sentiment_score'], reverse=True)[:5],
                team2.upper(): sorted(team2_details, key=lambda x: x['sentiment_score'], reverse=True)[:5]
            },
            'top_negative_comments': {
                team1.upper(): sorted(team1_details, key=lambda x: x['sentiment_score'])[:5],
                team2.upper(): sorted(team2_details, key=lambda x: x['sentiment_score'])[:5]
            }
        }

        # 결과 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(prediction_result, f, ensure_ascii=False, indent=2)

        # 콘솔 출력
        print("\n" + "=" * 70)
        print(f"경기 예측: {team1.upper()} vs {team2.upper()}")
        print("=" * 70)
        print(f"\n[{team1.upper()} 감성 분석]")
        print(f"  데이터 개수: {len(team1_scores)}개")
        print(f"  평균 감성점수: {team1_avg:.4f}")
        print(f"  긍정 비율: {team1_positive_rate:.2%}")

        print(f"\n[{team2.upper()} 감성 분석]")
        print(f"  데이터 개수: {len(team2_scores)}개")
        print(f"  평균 감성점수: {team2_avg:.4f}")
        print(f"  긍정 비율: {team2_positive_rate:.2%}")

        print(f"\n[예측 결과]")
        print(f"  감성 격차: {abs(sentiment_gap):.4f}")
        print(f"  예상 승자: {predicted_winner.upper()}")
        print(f"  승리 확률: {win_probability:.2%}")
        print(f"  신뢰도: {confidence_label} ({confidence:.2%})")

        print(f"\n결과가 {output_file}에 저장되었습니다.")
        print("=" * 70)

        return prediction_result

    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {sentiment_file}")
        return None
    except Exception as e:
        print(f"예측 중 오류 발생: {e}")
        return None


def batch_predict_matches(sentiment_file, matches, output_dir='predictions'):
    """
    여러 경기를 일괄 예측
    matches: [('gsw', 'hou'), ('lal', 'bos'), ...]
    """
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = []

    for team1, team2 in matches:
        output_file = f"{output_dir}/{team1}_vs_{team2}_prediction.json"
        result = predict_match_winner(sentiment_file, team1, team2, output_file)
        if result:
            results.append(result)

    # 전체 요약
    summary_file = f"{output_dir}/all_predictions_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n모든 예측 결과가 {output_dir} 폴더에 저장되었습니다.")


def main():
    """
    메인 실행 함수
    """
    input_filename = "final_data.json"
    output_filename = "final_score_model_sentiment.json"
    summary_filename = "team_sentiment_summary.json"

    print("=" * 60)
    print("ELECTRA 기반 감성 분석 시작")
    print("=" * 60)
    print()

    # 1. 감성 분석 수행
    process_json_with_model_sentiment(input_filename, output_filename)

    # 2. 팀별 요약 통계 계산
    #print("\n팀별 감성 요약 계산 중...")
    #calculate_team_sentiment_summary(output_filename, summary_filename)

    # 3. 경기 승패 예측 (예시)
    print("\n" + "=" * 60)
    print("경기 승패 예측")
    print("=" * 60)

    # 단일 경기 예측 예시
    #predict_match_winner(output_filename, 'mem', 'okc', 'prediction.json')

    # 여러 경기 일괄 예측 예시 (필요시 주석 해제)
    # matches = [
    #     ('gsw', 'hou'),
    #     ('lal', 'bos'),
    #     ('mia', 'den')
    # ]
    # batch_predict_matches(output_filename, matches)

    print("\n" + "=" * 60)
    print("모든 작업 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()