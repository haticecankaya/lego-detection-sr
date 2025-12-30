import streamlit as st
from PIL import Image
from src.config import PAGE_TITLE, LAYOUT
from src.model_utils import load_sr_model, load_detection_model, enhance_image
from src.utils import get_logger

logger = get_logger("StreamlitApp")

st.set_page_config(page_title=PAGE_TITLE, layout=LAYOUT)

def main():
    # --- SIDEBAR (YAN MENÜ) ---
    st.sidebar.title("⚙️ Ayarlar")
    
    st.sidebar.subheader("Görünüm Ayarları")
   
    main_img_width = st.sidebar.slider(
        "🖼️ Ana Resim Genişliği",
        min_value=300,
        max_value=1000,
        value=600,
        step=50,
        help="Yüklenen orijinal resmin ekrandaki boyutu."
    )
    
    crop_img_width = st.sidebar.slider(
        "🧩 Parça Resim Genişliği",
        min_value=100,
        max_value=600,
        value=300,
        step=50,
        help="Tespit edilen ve netleştirilen küçük resimlerin boyutu."
    )
    
    st.sidebar.info(
        "Bu uygulama, nesne tespiti için **YOLOv8** ve "
        "görüntü netleştirme için **RealESRGAN** kullanır."
    )

    # --- ANA SAYFA ---
    st.title(PAGE_TITLE)
    st.markdown("### 🔍 YOLOv8 ile Tespit ve ✨ RealESRGAN ile Netleştirme")

    with st.spinner("Yapay zeka modelleri yükleniyor..."):
        sr_upsampler = load_sr_model()
        det_model = load_detection_model()

    if not sr_upsampler or not det_model:
        st.error("🚨 Modeller yüklenemedi! Lütfen 'models/' klasörünü ve logları kontrol edin.")
        return

    
    uploaded_file = st.file_uploader("Analiz edilecek resmi yükleyin...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        
        original_image = Image.open(uploaded_file).convert("RGB")
        
        
        st.image(original_image, caption="Yüklenen Resim", width=main_img_width)

      
        if st.button("🚀 Analiz Et ve Netleştir", type="primary"):
            with st.spinner('Nesneler aranıyor ve görüntü işleniyor...'):
                logger.info("Analiz işlemi başlatıldı.")
                
                
                results = det_model(original_image)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                if len(boxes) == 0:
                    st.warning("⚠️ Resimde herhangi bir nesne tespit edilemedi. Güven eşiğini düşürmeyi deneyin.")
                    logger.warning("Nesne tespit edilemedi.")
                else:
                    st.success(f"✅ {len(boxes)} adet nesne bulundu.")
                    logger.info(f"{len(boxes)} nesne tespit edildi.")
                    
                    st.divider()
                    st.subheader("Sonuçlar")
                    
                    # Sonuçları Listele
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box
                        crop_img = original_image.crop((x1, y1, x2, y2))
                        
                       
                        enhanced_img = enhance_image(sr_upsampler, crop_img)
                        
                        
                        with st.container():
                            st.markdown(f"**Nesne #{i+1}**")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                               
                                st.image(crop_img, caption="Orijinal (Kırpılmış)", width=crop_img_width)
                            with col2:
                                
                                st.image(enhanced_img, caption="Netleştirilmiş (RealESRGAN)", width=crop_img_width)
                            
                            st.divider()

if __name__ == "__main__":
    main()