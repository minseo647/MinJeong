# 실행방법
#streamlit run "C:\Users\권민서\OneDrive\바탕 화면\MinJeong\parking_search_bot.py"

# 실행방법
#streamlit run "C:\Users\권민서\OneDrive\바탕 화면\MinJeong\parking_search_bot.py"

# 새로운 CSV 파일을 사용하는 주차장 검색 앱
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import math
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time
import re
from typing import Dict, Optional, List
import chardet

# 페이지 설정
st.set_page_config(
    page_title="서칭, 파킹",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# plotly 설치 확인 및 조건부 import
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly가 설치되지 않았습니다. matplotlib를 사용합니다.")

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Malgun Gothic', 'AppleGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ImprovedLocationSearch:
    def __init__(self, openai_api_key: str = ""):
        self.openai_api_key = openai_api_key
        
        # 확장된 정확한 좌표 데이터베이스
        self.location_database = {
            # 마포구 지역 (홍익대학교 근처)
            '상수역': {'latitude': 37.5475, 'longitude': 126.9127, 'name': '상수역 (마포구)', 'district': '마포구'},
            '상수': {'latitude': 37.5475, 'longitude': 126.9127, 'name': '상수역 (마포구)', 'district': '마포구'},
            '홍대입구역': {'latitude': 37.5563, 'longitude': 126.9234, 'name': '홍대입구역 (마포구)', 'district': '마포구'},
            '홍대': {'latitude': 37.5563, 'longitude': 126.9234, 'name': '홍대입구역 (마포구)', 'district': '마포구'},
            '홍익대학교': {'latitude': 37.5511, 'longitude': 126.9240, 'name': '홍익대학교 (마포구)', 'district': '마포구'},
            '홍익대': {'latitude': 37.5511, 'longitude': 126.9240, 'name': '홍익대학교 (마포구)', 'district': '마포구'},
            '합정역': {'latitude': 37.5496, 'longitude': 126.9138, 'name': '합정역 (마포구)', 'district': '마포구'},
            '망원역': {'latitude': 37.5557, 'longitude': 126.9105, 'name': '망원역 (마포구)', 'district': '마포구'},
            
            # 서대문구 지역
            '신촌역': {'latitude': 37.5597, 'longitude': 126.9423, 'name': '신촌역 (서대문구)', 'district': '서대문구'},
            '신촌': {'latitude': 37.5597, 'longitude': 126.9423, 'name': '신촌역 (서대문구)', 'district': '서대문구'},
            '이대역': {'latitude': 37.5563, 'longitude': 126.9456, 'name': '이대역 (서대문구)', 'district': '서대문구'},
            '연세대학교': {'latitude': 37.5665, 'longitude': 126.9387, 'name': '연세대학교 (서대문구)', 'district': '서대문구'},
            '이화여자대학교': {'latitude': 37.5563, 'longitude': 126.9468, 'name': '이화여자대학교 (서대문구)', 'district': '서대문구'},
            
            # 동작구 지역
            '숭실대학교': {'latitude': 37.4963, 'longitude': 126.9573, 'name': '숭실대학교 (동작구)', 'district': '동작구'},
            '숭실대': {'latitude': 37.4963, 'longitude': 126.9573, 'name': '숭실대학교 (동작구)', 'district': '동작구'},
            '상도역': {'latitude': 37.4972, 'longitude': 126.9533, 'name': '상도역 (동작구)', 'district': '동작구'},
            '사당역': {'latitude': 37.4767, 'longitude': 126.9813, 'name': '사당역 (동작구)', 'district': '동작구'},
            '총신대입구역': {'latitude': 37.4867, 'longitude': 126.9627, 'name': '총신대입구역 (동작구)', 'district': '동작구'},
            
            # 관악구 지역 (혼동 방지용)
            '서울대학교': {'latitude': 37.4601, 'longitude': 126.9520, 'name': '서울대학교 (관악구)', 'district': '관악구'},
            '서울대': {'latitude': 37.4601, 'longitude': 126.9520, 'name': '서울대학교 (관악구)', 'district': '관악구'},
            '서울대입구역': {'latitude': 37.4813, 'longitude': 126.9527, 'name': '서울대입구역 (관악구)', 'district': '관악구'},
            '신림역': {'latitude': 37.4842, 'longitude': 126.9292, 'name': '신림역 (관악구)', 'district': '관악구'},
            
            # 강남 지역
            '강남역': {'latitude': 37.4979, 'longitude': 127.0276, 'name': '강남역', 'district': '강남구'},
            '역삼역': {'latitude': 37.5003, 'longitude': 127.0360, 'name': '역삼역', 'district': '강남구'},
            '선릉역': {'latitude': 37.5045, 'longitude': 127.0491, 'name': '선릉역', 'district': '강남구'},
            
            # 기타 주요 지역
            '종로': {'latitude': 37.5709, 'longitude': 126.9928, 'name': '종로3가역', 'district': '종로구'},
            '명동': {'latitude': 37.5636, 'longitude': 126.9822, 'name': '명동역', 'district': '중구'},
        }
    
    def search_location(self, address: str) -> Dict:
        """다단계 위치 검색"""
        # 1단계: 정확한 데이터베이스 매칭
        exact_match = self._exact_database_search(address)
        if exact_match:
            return exact_match
        
        # 2단계: 부분 매칭
        partial_match = self._partial_database_search(address)
        if partial_match:
            return partial_match
        
        # 3단계: OpenAI 개선된 프롬프트
        openai_result = self._improved_openai_geocoding(address)
        if openai_result:
            return openai_result
        
        # 4단계: 기본값 (서울시청)
        return self._get_default_coordinates(address)
    
    def _exact_database_search(self, address: str) -> Optional[Dict]:
        """정확한 데이터베이스 검색"""
        address_clean = address.strip().lower()
        
        for key, coords in self.location_database.items():
            if key.lower() == address_clean:
                return {
                    'latitude': coords['latitude'],
                    'longitude': coords['longitude'],
                    'source': f"정확 매칭: {coords['name']}",
                    'confidence': 'high',
                    'district': coords.get('district', '서울시')
                }
        return None
    
    def _partial_database_search(self, address: str) -> Optional[Dict]:
        """부분 매칭 검색"""
        address_clean = address.strip().lower()
        matches = []
        
        for key, coords in self.location_database.items():
            key_lower = key.lower()
            
            if key_lower in address_clean or address_clean in key_lower:
                length_diff = abs(len(key_lower) - len(address_clean))
                score = 1.0 / (1 + length_diff * 0.1)
                
                matches.append({
                    'coords': coords,
                    'score': score,
                    'matched_key': key
                })
        
        if matches:
            best_match = max(matches, key=lambda x: x['score'])
            coords = best_match['coords']
            
            return {
                'latitude': coords['latitude'],
                'longitude': coords['longitude'],
                'source': f"부분 매칭: {coords['name']} (키워드: {best_match['matched_key']})",
                'confidence': 'medium',
                'district': coords.get('district', '서울시')
            }
        
        return None
    
    def _improved_openai_geocoding(self, address: str) -> Optional[Dict]:
        """개선된 OpenAI 지오코딩"""
        if not self.openai_api_key or self.openai_api_key.strip() == "":
            return None
        
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""당신은 서울특별시 전문 지오코딩 시스템입니다.

입력 위치: "{address}"

중요한 정확한 위치 정보:
- 홍익대학교: 37.5511, 126.9240 (마포구)
- 숭실대학교: 37.4963, 126.9573 (동작구) ← 정확히 동작구 상도동
- 서울대학교: 37.4601, 126.9520 (관악구)
- 상수역: 37.5475, 126.9127 (마포구)
- 홍대입구역: 37.5563, 126.9234 (마포구)  
- 신촌역: 37.5597, 126.9423 (서대문구)
- 강남역: 37.4979, 127.0276 (강남구)
- 상도역: 37.4972, 126.9533 (동작구)
- 사당역: 37.4767, 126.9813 (동작구)

주의사항:
- 숭실대학교는 절대 관악구가 아닙니다! 동작구입니다!
- 서울대학교가 관악구에 있습니다.

응답 형식:
위도: XX.XXXX
경도: XXX.XXXX
지역: OO구
신뢰도: HIGH/MEDIUM/LOW

서울시가 아니면 "INVALID"만 응답하세요."""
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "당신은 서울특별시 전문 지오코딩 AI입니다."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 150,
                "temperature": 0.0
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                if content == "INVALID":
                    return None
                
                lat_match = re.search(r'위도[:\s]*([0-9]{2}\.[0-9]+)', content)
                lng_match = re.search(r'경도[:\s]*([0-9]{3}\.[0-9]+)', content)
                region_match = re.search(r'지역[:\s]*([^\n]+)', content)
                confidence_match = re.search(r'신뢰도[:\s]*(HIGH|MEDIUM|LOW)', content)
                
                if lat_match and lng_match:
                    lat = float(lat_match.group(1))
                    lng = float(lng_match.group(1))
                    
                    if self._is_valid_seoul_coordinate(lat, lng):
                        region = region_match.group(1) if region_match else "서울시"
                        confidence = confidence_match.group(1) if confidence_match else "MEDIUM"
                        
                        return {
                            'latitude': lat,
                            'longitude': lng,
                            'source': f"OpenAI: {region} (신뢰도: {confidence})",
                            'confidence': confidence.lower(),
                            'district': region
                        }
            
        except Exception as e:
            print(f"OpenAI 지오코딩 오류: {e}")
        
        return None
    
    def _is_valid_seoul_coordinate(self, lat: float, lng: float) -> bool:
        """서울 좌표 범위 엄격 검증"""
        seoul_bounds = {
            'lat_min': 37.413,
            'lat_max': 37.701,
            'lng_min': 126.764,
            'lng_max': 127.183
        }
        
        return (seoul_bounds['lat_min'] <= lat <= seoul_bounds['lat_max'] and
                seoul_bounds['lng_min'] <= lng <= seoul_bounds['lng_max'])
    
    def _get_default_coordinates(self, address: str) -> Dict:
        """기본 좌표 반환"""
        return {
            'latitude': 37.5666805,
            'longitude': 126.9784147,
            'source': f"기본값: 서울시청 ('{address}' 검색 실패)",
            'confidence': 'low',
            'district': '중구'
        }

class ParkingSearchBot:
    def __init__(self):
        # 세션 상태 초기화
        if 'parking_data' not in st.session_state:
            st.session_state.parking_data = None
        if 'user_coords' not in st.session_state:
            st.session_state.user_coords = None
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        if 'user_location' not in st.session_state:
            st.session_state.user_location = ""
        if 'openai_api_key' not in st.session_state:
            st.session_state.openai_api_key = ""
        
        self.openai_api_key = st.session_state.openai_api_key
        
        # 여러 가능한 경로를 시도
        possible_paths = [
            r"C:\Users\권민서\OneDrive\바탕 화면\MinJeong\parking.csv",
            r"C:\Users\권민서\OneDrive\Desktop\MinJeong\parking.csv",
            r"parking.csv",
            os.path.join(os.path.dirname(__file__), "parking.csv")
        ]
        
        self.csv_file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                self.csv_file_path = path
                break
    
    def detect_encoding(self, file_path):
        """파일 인코딩 자동 감지"""
        try:
            with open(file_path, 'rb') as f:
                sample = f.read(10000)  # 처음 10KB 읽기
                detected = chardet.detect(sample)
                return detected['encoding']
        except:
            return None
    
    def load_data(self):
        """새로운 parking.csv 파일 로드"""
        try:
            if self.csv_file_path is None:
                st.error("❌ parking.csv 파일을 찾을 수 없습니다!")
                return False
            
            st.info(f"📂 파일 경로: {self.csv_file_path}")
            
            # 인코딩 감지
            detected_encoding = self.detect_encoding(self.csv_file_path)
            st.info(f"🔍 감지된 인코딩: {detected_encoding}")
            
            # 다양한 인코딩으로 시도
            encodings_to_try = [
                detected_encoding,
                'utf-8',
                'cp949',
                'euc-kr', 
                'cp1252',
                'iso-8859-1'
            ]
            
            parking_data = None
            successful_encoding = None
            
            for encoding in encodings_to_try:
                if encoding is None:
                    continue
                try:
                    st.write(f"🔄 {encoding} 인코딩으로 시도 중...")
                    parking_data = pd.read_csv(self.csv_file_path, encoding=encoding)
                    
                    # 한글이 제대로 읽혔는지 확인
                    columns = list(parking_data.columns)
                    if any('주차장' in str(col) for col in columns):
                        successful_encoding = encoding
                        st.success(f"✅ {encoding} 인코딩으로 성공!")
                        break
                        
                except Exception as e:
                    st.write(f"❌ {encoding} 실패: {str(e)[:100]}...")
                    continue
            
            if parking_data is None:
                st.error("❌ 모든 인코딩 시도 실패!")
                return False
            
            st.success(f"✅ 총 {len(parking_data)}개의 주차장 데이터 로드 완료 (인코딩: {successful_encoding})")
            
            # 컬럼 정보 표시
            st.info("📋 컬럼 정보:")
            st.write(f"컬럼 수: {len(parking_data.columns)}")
            st.write("주요 컬럼:", list(parking_data.columns)[:10])
            
            # 새 CSV 파일 구조에 맞게 데이터 정제
            # 필수 컬럼 확인
            required_columns = ['위도', '경도', '총 주차면', '주차장명']
            missing_columns = [col for col in required_columns if col not in parking_data.columns]
            
            if missing_columns:
                st.error(f"❌ 필수 컬럼이 없습니다: {missing_columns}")
                return False
            
            # 서울 지역만 필터링 (좌표 기반)
            seoul_bounds = {'lat_min': 37.4, 'lat_max': 37.7, 'lng_min': 126.8, 'lng_max': 127.2}
            
            seoul_data = parking_data[
                (parking_data['위도'] >= seoul_bounds['lat_min']) &
                (parking_data['위도'] <= seoul_bounds['lat_max']) &
                (parking_data['경도'] >= seoul_bounds['lng_min']) &
                (parking_data['경도'] <= seoul_bounds['lng_max'])
            ].copy()
            
            st.info(f"🌍 서울 지역 필터링 결과: {len(seoul_data)}개")
            
            if len(seoul_data) == 0:
                st.error("❌ 서울 지역 데이터가 없습니다!")
                return False
            
            # 데이터 정제
            seoul_data = seoul_data.dropna(subset=['경도', '위도', '총 주차면'])
            seoul_data['총 주차면'] = pd.to_numeric(seoul_data['총 주차면'], errors='coerce')
            seoul_data = seoul_data.dropna(subset=['총 주차면'])
            seoul_data = seoul_data[seoul_data['총 주차면'] > 0]
            
            st.info(f"📊 데이터 정제 후: {len(seoul_data)}개")
            
            # 새 컬럼명으로 매핑 (기존 코드와의 호환성을 위해)
            seoul_data = seoul_data.rename(columns={
                '총 주차면': '주차구획수',
                '기본 주차 요금': '기본요금',
                '추가 단위 요금': '추가요금',
                '기본 주차 시간(분 단위)': '기본시간',
                '추가 단위 시간(분 단위)': '추가시간',
                '주소': '주차장주소',
                '전화번호': '연락처',
                '주차장 종류명': '주차장구분',
                '운영구분명': '관리기관명'
            })
            
            # 시뮬레이션 데이터 생성
            np.random.seed(42)
            
            def safe_random_parked(total_spaces):
                try:
                    if total_spaces <= 0:
                        return 0
                    max_parked = max(1, int(total_spaces * 0.8))
                    return np.random.randint(0, max_parked + 1)
                except:
                    return 0
            
            seoul_data['현재주차수'] = seoul_data['주차구획수'].apply(safe_random_parked)
            
            def safe_hourly_rate(row):
                try:
                    # 기본요금이 있으면 그것을 기준으로 시간당 요금 계산
                    if pd.notna(row.get('기본요금', None)) and row.get('기본요금', 0) > 0:
                        base_rate = row['기본요금']
                        base_time = row.get('기본시간', 60)  # 기본 60분
                        if base_time > 0:
                            return int(base_rate * (60 / base_time))  # 시간당 요금으로 환산
                        else:
                            return int(base_rate)
                    
                    # 유무료 구분 확인
                    if '무료' in str(row.get('유무료구분명', '')) or row.get('유무료구분명', '') == '무료':
                        return 0
                    elif '공영' in str(row.get('주차장구분', '')):
                        return np.random.randint(1000, 3001)
                    else:
                        return np.random.randint(2000, 5001)
                except:
                    return 2000
            
            seoul_data['시간당요금'] = seoul_data.apply(safe_hourly_rate, axis=1)
            seoul_data = seoul_data[seoul_data['현재주차수'] >= 0]
            seoul_data = seoul_data[seoul_data['현재주차수'] <= seoul_data['주차구획수']]
            
            st.session_state.parking_data = seoul_data
            st.success(f"🚗 서울 지역 주차장 {len(seoul_data)}개 데이터 준비 완료")
            
            # 데이터 샘플 표시
            if len(seoul_data) > 0:
                st.info("📋 데이터 샘플:")
                sample_cols = ['주차장명', '주차구획수', '현재주차수', '시간당요금']
                available_cols = [col for col in sample_cols if col in seoul_data.columns]
                if available_cols:
                    sample_data = seoul_data[available_cols].head(3)
                    st.dataframe(sample_data)
            
            return True
            
        except Exception as e:
            st.error(f"❌ 데이터 로드 실패: {e}")
            st.info("💡 상세 오류 정보:")
            st.code(str(e))
            return False
    
    def geocode_with_openai(self, address):
        """개선된 지오코딩 메서드"""
        search_engine = ImprovedLocationSearch(self.openai_api_key)
        result = search_engine.search_location(address)
        
        confidence_colors = {
            'high': '🟢',
            'medium': '🟡', 
            'low': '🔴'
        }
        
        confidence = result.get('confidence', 'low')
        color = confidence_colors.get(confidence, '🔴')
        
        st.info(f"{color} **검색 결과**: {result['source']}")
        st.caption(f"📍 좌표: ({result['latitude']:.4f}, {result['longitude']:.4f})")
        
        return {
            'latitude': result['latitude'],
            'longitude': result['longitude']
        }
    
    def get_current_location_by_ip(self):
        """IP 기반으로 현재 위치 가져오기"""
        services = [
            {
                'url': 'https://ipapi.co/json/',
                'lat_key': 'latitude',
                'lng_key': 'longitude',
                'name': 'ipapi.co'
            },
            {
                'url': 'http://ip-api.com/json/',
                'lat_key': 'lat',
                'lng_key': 'lon',
                'name': 'ip-api.com'
            }
        ]
        
        for service in services:
            try:
                response = requests.get(service['url'], timeout=5)
                data = response.json()
                
                if service['lat_key'] in data and service['lng_key'] in data:
                    city = data.get('city', data.get('regionName', '알 수 없음'))
                    return {
                        'latitude': float(data[service['lat_key']]),
                        'longitude': float(data[service['lng_key']]),
                        'city': city,
                        'source': service['name']
                    }
                    
            except Exception as e:
                st.warning(f"{service['name']} 서비스 오류: {e}")
                continue
        
        return None
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """두 좌표 간 거리 계산 (Haversine formula)"""
        R = 6371
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def process_parking_data(self):
        """사용자 위치 기준으로 주차장 데이터 처리"""
        if st.session_state.user_coords is None or st.session_state.parking_data is None:
            return 0
        
        parking_data = st.session_state.parking_data.copy()
        parking_data['거리'] = parking_data.apply(
            lambda row: self.calculate_distance(
                st.session_state.user_coords['latitude'], st.session_state.user_coords['longitude'],
                row['위도'], row['경도']
            ), axis=1
        )
        
        processed_data = parking_data[parking_data['거리'] <= 5].copy()
        processed_data['잔여석'] = processed_data['주차구획수'] - processed_data['현재주차수']
        
        st.session_state.processed_data = processed_data
        return len(processed_data)

def main():
    app = ParkingSearchBot()
    
    # 사이드바
    with st.sidebar:
        st.title("⚙️ 설정")
        
        api_key = st.text_input(
            "🔑 OpenAI API 키",
            value=st.session_state.openai_api_key,
            type="password",
            help="더 정확한 위치 검색을 위해 API 키를 입력하세요."
        )
        
        if api_key != st.session_state.openai_api_key:
            st.session_state.openai_api_key = api_key
            app.openai_api_key = api_key
            st.success("✅ API 키가 저장되었습니다!")
        
        if st.session_state.openai_api_key:
            st.success("🔓 OpenAI API 연결됨")
        else:
            st.warning("🔒 기본 모드")
        
        st.markdown("---")
        
        # 새 CSV 파일 정보
        with st.expander("📊 새 데이터셋 정보"):
            st.markdown("""
            **새로운 주차장 데이터**:
            - 전체 주차장 수: 6,240개
            - 주요 필드: 주차장명, 주소, 총 주차면, 요금 정보
            - 운영 시간 정보 포함
            - 유무료 구분 정보
            """)
        
        st.info("💡 **사용법**\n1. 위치를 입력하세요\n2. 주차장 목록을 확인하세요\n3. 그래프로 분석하세요")
    
    # 메인 화면
    if st.session_state.parking_data is None:
        show_start_screen(app)
    elif st.session_state.user_coords is None:
        show_location_input(app)
    else:
        show_parking_dashboard(app)

def show_start_screen(app):
    """시작 화면"""
    st.markdown("""
    <style>
    .main-title {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        color: #1f1f1f;
        margin-bottom: 1rem;
    }
    .sub-title {
        font-size: 1.5rem;
        text-align: center;
        color: #555;
        margin-bottom: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-title">🚗 서칭, 파킹</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">쉽고 빠른 주변 주차장 검색 (새 데이터)</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 시작하기", type="primary", use_container_width=True):
            if app.load_data():
                st.rerun()

def show_location_input(app):
    """위치 입력 화면"""
    st.title("📍 위치 입력")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("📍 현재 위치 사용 (IP 기반)", type="secondary", use_container_width=True):
            with st.spinner("🌐 현재 위치를 찾는 중..."):
                location_data = app.get_current_location_by_ip()
                
                if location_data:
                    st.success(f"✅ 현재 위치: {location_data['city']}")
                    st.info(f"📊 좌표: {location_data['latitude']:.4f}, {location_data['longitude']:.4f}")
                    
                    st.session_state.user_coords = {
                        'latitude': location_data['latitude'],
                        'longitude': location_data['longitude']
                    }
                    st.session_state.user_location = f"현재 위치 ({location_data['city']})"
                    
                    count = app.process_parking_data()
                    st.success(f"🎯 현재 위치 기준 {count}개의 주차장을 찾았습니다!")
                    
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("❌ 현재 위치를 찾을 수 없습니다. 수동으로 입력해주세요.")
    
    with col2:
        st.markdown("**또는**")
    
    st.markdown("### 🖊️ 직접 입력")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        location = st.text_input(
            "검색할 위치를 입력하세요",
            placeholder="예: 홍익대학교, 상수역, 강남역"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_btn = st.button("🔍 검색", type="primary", use_container_width=True)
    
    st.markdown("**인기 지역:**")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    popular_locations = ["홍익대학교", "상수역", "강남역", "신촌", "숭실대학교"]
    for i, loc in enumerate(popular_locations):
        with [col1, col2, col3, col4, col5][i]:
            if st.button(loc, use_container_width=True):
                location = loc
                search_btn = True
    
    if search_btn and location:
        with st.spinner("🔍 위치를 분석하고 주차장 데이터를 처리중..."):
            coords = app.geocode_with_openai(location)
            st.session_state.user_coords = coords
            st.session_state.user_location = location
            
            count = app.process_parking_data()
            
            st.success(f"✅ '{location}' 검색 완료! {count}개의 주차장을 찾았습니다.")
            time.sleep(2)
            st.rerun()

def show_parking_dashboard(app):
    """주차장 대시보드"""
    st.title(f"🏢 {st.session_state.user_location} 주변 주차장")
    
    if st.session_state.processed_data is None or len(st.session_state.processed_data) == 0:
        st.warning("검색 결과가 없습니다. 다른 위치를 시도해보세요.")
        if st.button("🔄 새로 검색"):
            st.session_state.user_coords = None
            st.rerun()
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 검색 결과", f"{len(st.session_state.processed_data)}개")
    
    with col2:
        avg_distance = st.session_state.processed_data['거리'].mean()
        st.metric("📍 평균 거리", f"{avg_distance:.1f}km")
    
    with col3:
        total_spaces = st.session_state.processed_data['잔여석'].sum()
        st.metric("🅿️ 총 잔여석", f"{total_spaces}개")
    
    with col4:
        avg_rate = st.session_state.processed_data['시간당요금'].mean()
        st.metric("💰 평균 요금", f"{avg_rate:,.0f}원")
    
    tab1, tab2, tab3 = st.tabs(["📋 주차장 목록", "📊 그래프 분석", "🗺️ 지도"])
    
    with tab1:
        show_parking_list()
    
    with tab2:
        show_graph_analysis()
    
    with tab3:
        show_map_view()
    
    if st.button("🔄 새로 검색하기"):
        st.session_state.user_coords = None
        st.session_state.processed_data = None
        st.rerun()

def show_parking_list():
    """주차장 목록 표시"""
    col1, col2 = st.columns([1, 3])
    
    with col1:
        sort_option = st.selectbox("정렬 기준", ["거리순", "잔여석순", "요금순"])
    
    if sort_option == "거리순":
        sorted_data = st.session_state.processed_data.sort_values('거리')
    elif sort_option == "잔여석순":
        sorted_data = st.session_state.processed_data.sort_values('잔여석', ascending=False)
    else:
        sorted_data = st.session_state.processed_data.sort_values('시간당요금')
    
    for idx, (_, row) in enumerate(sorted_data.head(20).iterrows()):
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.subheader(f"🏢 {row['주차장명']}")
                # 새 CSV 구조에 맞게 주소 표시
                address = row.get('주차장주소', row.get('주소', 'N/A'))
                st.caption(f"📍 {address}")
                
                # 주차장 종류 및 운영 정보 표시
                parking_type = row.get('주차장구분', 'N/A')
                operation_type = row.get('관리기관명', 'N/A')
                st.caption(f"🏛️ {parking_type} | 🏢 {operation_type}")
                
                # 연락처 정보
                contact = row.get('연락처', 'N/A')
                if contact and contact != 'N/A':
                    st.caption(f"📞 {contact}")
            
            with col2:
                st.metric("거리", f"{row['거리']:.1f}km")
            
            with col3:
                available_color = "🟢" if row['잔여석'] > 20 else "🟡" if row['잔여석'] > 5 else "🔴"
                st.metric("잔여석", f"{available_color} {row['잔여석']}/{row['주차구획수']}")
            
            with col4:
                if row['시간당요금'] == 0:
                    st.metric("요금", "🆓 무료")
                else:
                    st.metric("시간당 요금", f"💰 {row['시간당요금']:,}원")
            
            # 운영 시간 정보 추가
            weekday_start = row.get('평일 운영 시작시각(HHMM)', None)
            weekday_end = row.get('평일 운영 종료시각(HHMM)', None)
            
            if pd.notna(weekday_start) and pd.notna(weekday_end):
                start_time = f"{int(weekday_start):04d}"[:2] + ":" + f"{int(weekday_start):04d}"[2:]
                end_time = f"{int(weekday_end):04d}"[:2] + ":" + f"{int(weekday_end):04d}"[2:]
                st.caption(f"🕐 평일 운영: {start_time} ~ {end_time}")
            
            st.markdown("---")

def show_graph_analysis():
    """그래프 분석 화면"""
    st.subheader("📊 주차장 분석 그래프")
    
    y_axis = st.radio("분석할 데이터를 선택하세요:", ["잔여석", "시간당요금"], horizontal=True)
    
    top_15 = st.session_state.processed_data.nsmallest(15, '거리')
    
    if PLOTLY_AVAILABLE:
        fig = px.scatter(
            top_15,
            x='거리',
            y=y_axis,
            hover_name='주차장명',
            hover_data={'주차구획수': True, '주차장구분': True},
            title=f"거리 vs {y_axis}",
            labels={
                '거리': '거리 (km)',
                '잔여석': '잔여석 (개)',
                '시간당요금': '시간당 요금 (원)'
            },
            color=y_axis,
            size='주차구획수',
            size_max=15
        )
        
        fig.update_layout(height=500, showlegend=True, font=dict(family="Arial", size=12))
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_data = top_15['거리']
        y_data = top_15[y_axis]
        colors = top_15['주차구획수']
        
        scatter = ax.scatter(x_data, y_data, c=colors, s=100, alpha=0.7, cmap='viridis')
        
        for i, row in top_15.iterrows():
            name = row['주차장명'][:10] + '...' if len(row['주차장명']) > 10 else row['주차장명']
            ax.annotate(name, (row['거리'], row[y_axis]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('거리 (km)')
        ax.set_ylabel('잔여석 (개)' if y_axis == '잔여석' else '시간당 요금 (원)')
        ax.set_title(f'거리 vs {y_axis}')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('주차구획수')
        
        st.pyplot(fig)
    
    st.subheader("🔍 분석 인사이트")
    col1, col2 = st.columns(2)
    
    with col1:
        nearest = top_15.nsmallest(1, '거리').iloc[0]
        st.info(f"**가장 가까운 주차장**\n🏢 {nearest['주차장명']}\n📍 {nearest['거리']:.1f}km")
    
    with col2:
        if y_axis == "잔여석":
            most_available = top_15.nlargest(1, '잔여석').iloc[0]
            st.success(f"**잔여석이 가장 많은 곳**\n🏢 {most_available['주차장명']}\n🅿️ {most_available['잔여석']}개")
        else:
            cheapest = top_15.nsmallest(1, '시간당요금').iloc[0]
            rate_text = "무료" if cheapest['시간당요금'] == 0 else f"{cheapest['시간당요금']:,}원"
            st.success(f"**가장 저렴한 곳**\n🏢 {cheapest['주차장명']}\n💰 {rate_text}")

def show_map_view():
    """지도 뷰"""
    st.subheader("🗺️ 주차장 위치")
    
    if st.session_state.processed_data is None or len(st.session_state.processed_data) == 0:
        st.warning("표시할 주차장이 없습니다.")
        return
    
    map_data = st.session_state.processed_data.head(20).copy()
    
    st.map(map_data, latitude='위도', longitude='경도', size='주차구획수')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🟢 **여유** (20개 이상)")
    with col2:
        st.markdown("🟡 **보통** (6-20개)")
    with col3:
        st.markdown("🔴 **부족** (5개 이하)")

if __name__ == "__main__":
    main()