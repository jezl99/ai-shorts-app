import static_ffmpeg
static_ffmpeg.add_paths()

import streamlit as st
import os
import random
import cv2
import tempfile
import numpy as np
import json
import gc
import time
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, CompositeVideoClip, vfx
import whisper
import openai
import google.generativeai as genai

# -----------------------------------------------------------------------------
# 1. КОНФИГУРАЦИЯ СТРАНИЦЫ
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Shorts Studio Pro", 
    page_icon="🎬", 
    layout="wide"
)

st.title("🎬 AI Shorts & Reels Studio Pro")
st.caption("Превращайте горизонтальные видео в **вирусные 9:16 ролики** с трендовыми субтитрами и ИИ-анализом.")

# -----------------------------------------------------------------------------
# 2. КЭШИРОВАНИЕ МОДЕЛЕЙ
# -----------------------------------------------------------------------------
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

# -----------------------------------------------------------------------------
# 3. ФУНКЦИИ ИИ
# -----------------------------------------------------------------------------
def get_gemini_best_moments(transcript_text, count, duration, api_key):
    """Анализ текста через бесплатный Google Gemini"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config={"response_mime_type": "application/json"}
        )
        prompt = f"""
        У меня есть расшифровка видео. Найди {count} лучших, самых вирусных или интригующих моментов длительностью примерно {duration} секунд каждый.
        Верни ответ СТРОГО в формате JSON с ключом "moments", где каждый элемент имеет ключи "start" и "end" в секундах.
        Пример: {{"moments": [{"start": 12, "end": 27}, {"start": 85, "end": 100}]}}

        Текст видео:
        {transcript_text[:10000]}
        """
        response = model.generate_content(prompt)
        data = json.loads(response.text)
        return data.get("moments", data.get("clips", []))
    except Exception as e:
        st.warning(f"⚠️ Ошибка Gemini ({e}). Переключаюсь на случайную нарезку.")
        return []

def get_openai_best_moments(transcript_text, count, duration, api_key):
    """Анализ текста через OpenAI GPT-4o"""
    try:
        client = openai.OpenAI(api_key=api_key)
        prompt = f"""
        Найди {count} лучших моментов длительностью {duration} сек каждый.
        Верни JSON со списком объектов "start" и "end" в секундах.
        Текст: {transcript_text[:4000]}
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("moments", data.get("clips", []))
    except Exception as e:
        st.warning(f"⚠️ Ошибка OpenAI ({e}). Переключаюсь на случайную нарезку.")
        return []

# -----------------------------------------------------------------------------
# 4. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# -----------------------------------------------------------------------------
def draw_trendy_subtitles(frame, text, font_size, text_color, stroke_color, y_percent):
    if not text:
        return frame

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    w, h = img.size

    try:
        font = ImageFont.truetype("ariblk.ttf", font_size)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (w - text_w) // 2
    y = int(h * (y_percent / 100.0))

    draw.text(
        (x, y), 
        text, 
        font=font, 
        fill=text_color, 
        stroke_width=max(4, font_size // 10), 
        stroke_fill=stroke_color
    )

    return np.array(img)

def make_vertical_blurred(clip, target_w=1080, target_h=1920):
    bg = clip.resize(height=target_h)
    if bg.w < target_w:
        bg = bg.resize(width=target_w)
    
    bg = bg.crop(x_center=bg.w/2, y_center=bg.h/2, width=target_w, height=target_h)
    bg = bg.fl_image(lambda frame: cv2.GaussianBlur(frame, (91, 91), 0))
    bg = bg.fx(vfx.colorx, 0.5)
    
    fg = clip.resize(width=target_w)
    return CompositeVideoClip([bg, fg.set_position("center")])

def apply_uniqueness_effects(clip):
    speed_factor = random.uniform(0.98, 1.02)
    clip = clip.fx(vfx.speedx, speed_factor)
    
    brightness = random.uniform(0.97, 1.03)
    clip = clip.fx(vfx.colorx, brightness)
    
    if random.choice([True, False]):
        clip = clip.fx(vfx.resize, lambda t: 1 + 0.02 * (t / clip.duration))
        
    return clip

# -----------------------------------------------------------------------------
# 5. ИНТЕРФЕЙС И НАСТРОЙКИ
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 1. Параметры роликов")
    clip_count = st.number_input("Количество Shorts", min_value=1, max_value=100, value=5, step=1)
    clip_duration = st.slider("Длительность (сек)", 5, 60, 15)
    
    st.divider()
    st.header("🎨 2. Трендовые субтитры")
    enable_subs = st.checkbox("Включить динамические субтитры", value=True)
    
    if enable_subs:
        sub_y_percent = st.slider("Высота текста (% от верха)", 30, 85, 65)
        font_size = st.slider("Размер шрифта", 40, 100, 65)
        text_color = st.color_picker("Цвет текста", "#FFE600")
        stroke_color = st.color_picker("Цвет обводки", "#000000")
    
    st.divider()
    st.header("🤖 3. Умный ИИ-поиск")
    ai_provider = st.selectbox("Провайдер ИИ", ["Google Gemini (Бесплатно)", "OpenAI (GPT-4o)"])
    use_ai = st.checkbox("Включить умный поиск моментов", value=True)
    
    api_key = ""
    if use_ai:
        placeholder_text = "AIzaSy..." if "Gemini" in ai_provider else "sk-..."
        api_key = st.text_input(f"Ключ {ai_provider}", type="password", placeholder=placeholder_text)

# -----------------------------------------------------------------------------
# 6. ОСНОВНАЯ ЛОГИКА
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("📂 Перетащите сюда горизонтальное видео (MP4, MOV)", type=["mp4", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    if st.button("🚀 Начать генерацию роликов", type="primary", use_container_width=True):
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_in:
            tmp_in.write(uploaded_file.read())
            input_path = tmp_in.name
            
        output_dir = "generated_shorts"
        os.makedirs(output_dir, exist_ok=True)
        
        full_clip = None
        try:
            full_clip = VideoFileClip(input_path)
            total_duration = full_clip.duration
            
            transcript_segments = []
            
            if enable_subs or (use_ai and api_key):
                if full_clip.audio is None:
                    st.warning("⚠️ В видео не найден аудиопоток.")
                else:
                    status_text.text("🧠 ИИ извлекает и распознаёт речь...")
                    whisper_model = load_whisper_model()
                    
                    temp_main_audio = os.path.join(output_dir, "temp_full_audio.wav")
                    
                    full_clip.audio.write_audiofile(
                        temp_main_audio, 
                        fps=16000, 
                        nbytes=2, 
                        codec='pcm_s16le', 
                        logger=None
                    )
                    
                    if os.path.exists(temp_main_audio) and os.path.getsize(temp_main_audio) > 1000:
                        res = whisper_model.transcribe(temp_main_audio, language="ru")
                        transcript_segments = res.get("segments", [])
                        
                        try:
                            os.remove(temp_main_audio)
                        except:
                            pass

            ai_moments = []
            if use_ai and api_key and transcript_segments:
                status_text.text(f"🤖 {ai_provider} ищет самые вирусные фрагменты...")
                full_text = " ".join([s["text"] for s in transcript_segments])
                
                if "Gemini" in ai_provider:
                    ai_moments = get_gemini_best_moments(full_text, clip_count, clip_duration, api_key)
                else:
                    ai_moments = get_openai_best_moments(full_text, clip_count, clip_duration, api_key)

            for i in range(1, clip_count + 1):
                status_text.text(f"🎬 Рендеринг ролика {i} из {clip_count}...")
                
                if ai_moments and i <= len(ai_moments):
                    start_time = float(ai_moments[i-1]["start"])
                    end_time = float(ai_moments[i-1]["end"])
                else:
                    max_start = max(0, total_duration - clip_duration)
                    start_time = random.uniform(0, max_start)
                    end_time = min(start_time + clip_duration, total_duration)

                subclip = full_clip.subclip(start_time, end_time)
                vertical_clip = make_vertical_blurred(subclip)
                unique_clip = apply_uniqueness_effects(vertical_clip)
                
                if enable_subs and transcript_segments:
                    def subtitle_filter(get_frame, t_abs):
                        current_t = start_time + t_abs 
                        frame = get_frame(t_abs)
                        text_to_show = ""
                        
                        for seg in transcript_segments:
                            if seg["start"] <= current_t <= seg["end"]:
                                words = seg["text"].strip().upper().split()
                                text_to_show = " ".join(words[:3]) 
                                break
                                
                        return draw_trendy_subtitles(
                            frame, text_to_show, 
                            font_size, text_color, stroke_color, sub_y_percent
                        )
                        
                    unique_clip = unique_clip.fl(subtitle_filter)

                output_filename = os.path.join(output_dir, f"short_{i:03d}.mp4")
                
                unique_clip.write_videofile(
                    output_filename,
                    codec="libx264",
                    audio_codec="aac",
                    fps=30,
                    preset="ultrafast",
                    logger=None,
                    threads=4
                )
                
                subclip.close()
                vertical_clip.close()
                unique_clip.close()
                
                progress_bar.progress(i / clip_count)
            
            status_text.success(f"✅ Готово! Все {clip_count} видео сохранены в папку: {os.path.abspath(output_dir)}")
            st.balloons()

        except Exception as e:
            st.error(f"❌ Произошла ошибка при обработке: {e}")
            
        finally:
            if full_clip:
                full_clip.close()
            
            del full_clip
            gc.collect()
            time.sleep(0.5)
            
            try:
                if os.path.exists(input_path):
                    os.remove(input_path)
            except Exception:
                pass