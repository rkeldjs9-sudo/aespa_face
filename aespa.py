import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import glob
from pathlib import Path
import pandas as pd 
from scipy.spatial.distance import euclidean

# ----------------------------------------------------
# 1. 경로 상수 정의 (로컬 환경 자동 설정)
# ----------------------------------------------------
# 현재 스크립트 파일이 위치한 폴더를 BASE_PATH로 설정합니다. (aespa 폴더)
BASE_PATH = Path(os.path.dirname(os.path.abspath(__file__))) 
MODEL_PATH = BASE_PATH / "facenet_keras.h5"
DATA_PATH = BASE_PATH / "aespa_photo" 
MEMBER_NAMES = ["karina", "giselle", "winter", "ningning"]
TARGET_SIZE = (160, 160) 
DISTANCE_THRESHOLD = 1.5 

# ----------------------------------------------------
# 2. FaceNet 모델 로드 (st.cache_resource)
# ----------------------------------------------------
@st.cache_resource
def load_facenet_model():
    """FaceNet 모델을 로드하고 캐시합니다."""
    st.info(f"⏳ FaceNet 모델 로드 중: {MODEL_PATH}")
    try:
        if not MODEL_PATH.exists():
            st.error(f"❌ 모델 로드 오류: 'aespa' 폴더에 {MODEL_PATH.name} 파일을 찾을 수 없습니다.")
            st.stop()
        
        model = load_model(MODEL_PATH)
        st.success("✅ FaceNet 모델 로드 완료!")
        return model
    except Exception as e:
        st.error(f"❌ 모델 로드 중 예기치 않은 오류 발생: {e}")
        return None

# ----------------------------------------------------
# 3. 이미지 전처리 및 임베딩 함수 (핵심 FaceNet 로직)
# ----------------------------------------------------
def prewhiten(x):
    """FaceNet에 맞게 이미지를 전처리"""
    if x.ndim == 4:
        for i in range(x.shape[0]):
            x[i] = (x[i] - x[i].mean()) / np.maximum(np.std(x[i]), 1e-5)
        return x
    return (x - x.mean()) / np.maximum(np.std(x), 1e-5)

def get_face_from_image(image_data):
    """업로드된 이미지에서 얼굴을 추출하고 FaceNet 입력 크기로 조정"""
    try:
        image_np = np.frombuffer(image_data.read(), np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        face = face.astype('float64')
        face = prewhiten(face)
        face = np.expand_dims(face, axis=0)
        return face
    except Exception:
        return None
    
def get_face_from_path(image_path):
    """경로에서 이미지를 로드하여 전처리"""
    try:
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        face = face.astype('float64')
        face = prewhiten(face)
        face = np.expand_dims(face, axis=0)
        return face
    except Exception:
        return None

def get_embedding(model, face_array):
    """128차원 임베딩 벡터 추출"""
    embedding = model.predict(face_array, verbose=0)
    return embedding[0]

# ----------------------------------------------------
# 4. 유사도 계산 로직 (FR-1)
# ----------------------------------------------------
def calculate_similarity(distance):
    """유클리디안 거리를 0~100% 유사도 값으로 환산합니다."""
    # 공식: 유사도 (%) = max( 0, 100 - (D / 1.5) * 100 )
    similarity = np.maximum(0, 100 - (distance / DISTANCE_THRESHOLD) * 100)
    return similarity

# ----------------------------------------------------
# 5. 학습 데이터 임베딩 계산 (st.cache_data)
# ----------------------------------------------------
@st.cache_data
def load_and_preprocess_member_data(model):
    """멤버별 모든 사진을 로드하고 임베딩 벡터를 미리 계산하여 캐시합니다."""
    all_member_data = {}
    st.info("⏳ 학습 데이터 로드 및 임베딩 계산 중... (처음 1회만 실행)")
    
    progress_bar = st.progress(0, text="데이터 처리 중...")
    
    for i, member_name in enumerate(MEMBER_NAMES):
        member_folder = DATA_PATH / member_name
        
        if not member_folder.exists():
            st.warning(f"⚠️ 경고: 멤버 폴더를 찾을 수 없습니다: {member_folder}")
            continue

        image_files = sorted(list(member_folder.glob("*.jpg")) + list(member_folder.glob("*.png")))
        
        member_embeddings = []
        file_paths = []
        
        for file_path in image_files:
            face_array = get_face_from_path(file_path)
            if face_array is not None:
                embedding = get_embedding(model, face_array)
                member_embeddings.append(embedding)
                file_paths.append(file_path)

        if member_embeddings:
             all_member_data[member_name.upper()] = {
                "embeddings": np.array(member_embeddings),
                "paths": file_paths
            }
        
        progress_bar.progress((i + 1) / len(MEMBER_NAMES), text=f"데이터 처리 중: {member_name.upper()} 완료")
        
    progress_bar.empty()
    if all_member_data:
        st.success("✅ 학습 데이터 임베딩 계산 완료!")
    return all_member_data

# ----------------------------------------------------
# 6. Streamlit UI 및 메인 로직
# ----------------------------------------------------
st.set_page_config(
    page_title="aespa - 유사도 분석기",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("아이돌 닮은꼴 유사도 분석기 🔎")
st.subheader("FaceNet 기반의 인물 유사도 비교 시스템")

# FaceNet 모델 로드
facenet_model = load_facenet_model()

# 학습 데이터 로드 및 임베딩
member_embeddings_data = load_and_preprocess_member_data(facenet_model)

if not member_embeddings_data:
    st.error(f"❌ 분석을 위한 멤버 학습 데이터를 찾을 수 없습니다. 'aespa/aespa_photo/' 경로를 확인해 주세요.")
    st.stop()


uploaded_file = st.file_uploader("사진을 업로드하세요.", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.info("사진 업로드 완료! 분석을 시작합니다...")
    
    user_face_for_display = uploaded_file.getvalue()
    user_face = get_face_from_image(uploaded_file)
    
    if user_face is not None and facenet_model is not None:
        user_embedding = get_embedding(facenet_model, user_face)
        st.success("✅ 사용자 사진 임베딩 벡터 추출 완료!")

        # ----------------------------------------------------
        # 8. 유사도 분석 및 결과 정리 (핵심 로직)
        # ----------------------------------------------------
        
        analysis_results = []
        max_overall_similarity = -1.0
        best_match_data = {"member": "", "similarity": 0.0, "path": ""}

        # 멤버별 분석
        for member_name, data in member_embeddings_data.items():
            member_embeddings = data["embeddings"]
            member_paths = data["paths"]

            # 유클리디안 거리 계산
            distances = np.sqrt(np.sum((member_embeddings - user_embedding) ** 2, axis=1))
            
            # 유사도 변환 (FR-1)
            similarities = calculate_similarity(distances)
            
            # 최대 유사도 (Max P) 및 해당 사진 찾기 (FR-2)
            max_similarity_index = np.argmax(similarities)
            max_similarity = similarities[max_similarity_index]
            max_similarity_path = member_paths[max_similarity_index]
            
            # 평균 유사도 (Avg P)
            avg_similarity = np.mean(similarities)
            
            # 전체 최고 닮은꼴 업데이트 (FR-2)
            if max_similarity > max_overall_similarity:
                max_overall_similarity = max_similarity
                best_match_data["member"] = member_name
                best_match_data["similarity"] = max_similarity
                best_match_data["path"] = max_similarity_path

            # 테이블 결과 저장 (FR-3)
            analysis_results.append({
                "멤버": member_name,
                "최대 유사도 (%)": f"{max_similarity:.2f}%",
                "평균 유사도 (%)": f"{avg_similarity:.2f}%"
            })


        # ----------------------------------------------------
        # 9. 결과 출력 (FR-2, FR-3)
        # ----------------------------------------------------
        
        st.markdown("---")
        
        # FR-2: 전체 최고 닮은꼴 강조 출력
        st.markdown("## ✨ 최고 닮은꼴 분석 결과")
        st.markdown(
            f"**<span style='color:red; font-size:36px; font-weight:bold;'>{best_match_data['member']} ({best_match_data['similarity']:.2f}%)</span>**", 
            unsafe_allow_html=True
        )
        st.write("")


        # FR-2: 최고 유사 사진 출력 및 강조
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(user_face_for_display, caption="[업로드] 사용자 사진", use_column_width=True)

        with col2:
            try:
                best_match_image = cv2.imread(str(best_match_data['path']))
                best_match_image = cv2.cvtColor(best_match_image, cv2.COLOR_BGR2RGB)
                st.image(best_match_image, caption=f"[최고 매칭] {best_match_data['member']}의 사진", use_column_width=True)
            except Exception:
                st.error("❌ 최고 매칭 사진을 로드할 수 없습니다.")
        
        st.markdown("---")
        
        # FR-3: 멤버별 상세 비교 테이블
        st.markdown("## 📊 멤버별 상세 비교 테이블")
        results_df = pd.DataFrame(analysis_results)
        st.dataframe(results_df, use_container_width=True)
        
        
    
else:
    st.info("얼굴 인식을 위한 사진(JPG, PNG)을 업로드해주세요.")