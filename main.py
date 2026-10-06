import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# Load model only once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_model.keras")


model = load_model()


# TensorFlow Model Prediction
def model_prediction(test_image):
    # Open uploaded image
    image = Image.open(test_image).convert("RGB")

    # Resize image
    image = image.resize((128, 128))

    # Convert image to numpy array
    input_arr = np.array(image)

    # Convert single image to batch
    input_arr = np.expand_dims(input_arr, axis=0)

    # Prediction
    predictions = model.predict(input_arr, verbose=0)

    # Return index of maximum prediction AND its confidence score
    result_index = np.argmax(predictions)
    confidence = float(np.max(predictions))
    return result_index, confidence


def is_likely_plant_image(uploaded_file):
    """
    Basic heuristic to check if an image likely contains a plant/leaf.
    Checks if a minimum percentage of pixels have green as the dominant
    color channel, which is typical for plant/leaf images.
    Returns True if the image appears to be a plant, False otherwise.
    """
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((128, 128))  # Resize for faster processing
    img_array = np.array(image, dtype=np.float32)

    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

    # A pixel is "green-ish" if the green channel is notably higher than
    # both red and blue, or the image has earthy/plant tones
    green_dominant = (g > r) & (g > b) & (g > 40)
    brown_green = (g > 40) & (r > 30) & (b < r) & (g > b)  # Earthy plant tones
    plant_pixels = green_dominant | brown_green

    plant_ratio = np.sum(plant_pixels) / plant_pixels.size

    # If at least 8% of pixels look plant-like, consider it a plant image
    # This is a lenient threshold to avoid false rejections
    return plant_ratio > 0.08


# Class names
class_name = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]


# Disease information: description + solutions for each class
disease_info = {
    'Apple___Apple_scab': {
        'cause': 'Caused by the fungus **Venturia inaequalis**. It thrives in cool, wet weather during spring and spreads through wind-blown spores from infected fallen leaves.',
        'symptoms': 'Olive-green to dark brown velvety spots on leaves and fruit. Leaves may curl, yellow, and drop early. Fruit develops scabby, cracked lesions.',
        'solution': [
            'Apply fungicides such as captan or myclobutanil at bud break and continue through wet periods.',
            'Remove and destroy fallen infected leaves in autumn to reduce spore sources.',
            'Plant scab-resistant apple varieties (e.g., Liberty, Enterprise, Freedom).',
            'Prune trees to improve air circulation and sunlight penetration.',
            'Avoid overhead irrigation to keep foliage dry.',
        ]
    },
    'Apple___Black_rot': {
        'cause': 'Caused by the fungus **Botryosphaeria obtusa**. It overwinters in mummified fruit, dead bark, and cankers, spreading via rain splash.',
        'symptoms': 'Purple-bordered brown spots on leaves ("frog-eye" lesions). Fruit develops expanding brown-to-black rot. Cankers appear on branches.',
        'solution': [
            'Prune out all dead wood, cankers, and mummified fruit during dormant season.',
            'Apply fungicides (captan or thiophanate-methyl) from bloom through fruit development.',
            'Remove and destroy all infected plant debris from the orchard floor.',
            'Maintain tree vigor through proper fertilization and watering.',
            'Avoid wounding the bark during cultivation.',
        ]
    },
    'Apple___Cedar_apple_rust': {
        'cause': 'Caused by the fungus **Gymnosporangium juniperi-virginianae**. It requires two hosts to complete its life cycle: apple/crabapple and eastern red cedar (juniper).',
        'symptoms': 'Bright yellow-orange spots on upper leaf surfaces with tiny black dots. Tube-like structures appear on leaf undersides. Fruit may also develop lesions.',
        'solution': [
            'Remove nearby juniper/red cedar trees within a 2-mile radius if possible.',
            'Apply fungicides (myclobutanil or mancozeb) starting at pink bud stage.',
            'Plant rust-resistant apple varieties (e.g., Redfree, Liberty, Freedom).',
            'Scout juniper trees for galls in early spring and remove them before they release spores.',
            'Maintain good tree health with balanced fertilization.',
        ]
    },
    'Apple___healthy': {
        'cause': 'No disease detected.',
        'symptoms': 'The plant appears healthy with no visible signs of infection.',
        'solution': [
            'Continue regular monitoring for early signs of disease.',
            'Maintain proper watering, fertilization, and pruning schedules.',
            'Practice crop rotation and orchard sanitation.',
        ]
    },
    'Blueberry___healthy': {
        'cause': 'No disease detected.',
        'symptoms': 'The plant appears healthy with no visible signs of infection.',
        'solution': [
            'Continue regular monitoring for early signs of disease.',
            'Maintain acidic soil (pH 4.5–5.5) and proper mulching.',
            'Prune old canes to encourage new growth.',
        ]
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'cause': 'Caused by the fungus **Podosphaera clandestina**. It thrives in warm, dry weather with cool nights and high humidity.',
        'symptoms': 'White powdery coating on leaves, shoots, and sometimes fruit. Affected leaves may curl, turn yellow, and drop prematurely.',
        'solution': [
            'Apply sulfur-based or potassium bicarbonate fungicides at first sign of infection.',
            'Prune trees to improve air circulation within the canopy.',
            'Avoid excessive nitrogen fertilization which promotes susceptible new growth.',
            'Water at the base of the tree, avoiding overhead irrigation.',
            'Remove and destroy infected plant material.',
        ]
    },
    'Cherry_(including_sour)___healthy': {
        'cause': 'No disease detected.',
        'symptoms': 'The plant appears healthy with no visible signs of infection.',
        'solution': [
            'Continue regular monitoring for early signs of disease.',
            'Maintain proper watering and pruning schedules.',
            'Apply dormant sprays in late winter as preventive care.',
        ]
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'cause': 'Caused by the fungus **Cercospora zeae-maydis**. It thrives in warm, humid conditions and survives in crop residue from previous seasons.',
        'symptoms': 'Rectangular gray-to-tan lesions that run parallel to leaf veins. Lesions may merge, causing large areas of dead leaf tissue.',
        'solution': [
            'Plant resistant or tolerant corn hybrids.',
            'Practice crop rotation (rotate with non-host crops like soybeans).',
            'Till under crop residue after harvest to reduce fungal inoculum.',
            'Apply foliar fungicides (strobilurin or triazole-based) if disease pressure is high.',
            'Avoid planting corn in fields with heavy previous-year corn residue.',
        ]
    },
    'Corn_(maize)___Common_rust_': {
        'cause': 'Caused by the fungus **Puccinia sorghi**. Spores are wind-blown from southern regions and infect corn during cool, humid weather.',
        'symptoms': 'Small, round-to-elongate reddish-brown pustules on both leaf surfaces. Pustules release powdery rust-colored spores when rubbed.',
        'solution': [
            'Plant rust-resistant corn hybrids.',
            'Apply foliar fungicides (triazoles or strobilurins) if infection is severe and occurs before tasseling.',
            'Scout fields regularly during cool, humid weather.',
            'Early planting can help corn mature before peak rust season.',
            'No tillage adjustment needed as spores are wind-blown, not residue-borne.',
        ]
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'cause': 'Caused by the fungus **Exserohilum turcicum** (syn. Setosphaeria turcica). It survives in infected crop debris and spreads via wind-blown spores in wet conditions.',
        'symptoms': 'Long, elliptical gray-green to tan lesions (1–6 inches) on leaves. Severe infection can cause premature leaf death and significant yield loss.',
        'solution': [
            'Plant resistant corn hybrids with Ht genes.',
            'Practice crop rotation to reduce residue-borne inoculum.',
            'Apply fungicides (azoxystrobin, pyraclostrobin) at early tassel if disease appears.',
            'Till crop residue to reduce overwintering spores.',
            'Scout fields regularly, especially during wet seasons.',
        ]
    },
    'Corn_(maize)___healthy': {
        'cause': 'No disease detected.',
        'symptoms': 'The plant appears healthy with no visible signs of infection.',
        'solution': [
            'Continue regular monitoring for early signs of disease.',
            'Maintain proper irrigation and nutrient management.',
            'Practice crop rotation to prevent future disease buildup.',
        ]
    },
    'Grape___Black_rot': {
        'cause': 'Caused by the fungus **Guignardia bidwellii**. It overwinters in mummified berries and infected canes, spreading via rain splash in warm, wet weather.',
        'symptoms': 'Tan circular spots with dark borders on leaves. Fruit develops brown lesions that enlarge rapidly, turning berries into hard, black, shriveled mummies.',
        'solution': [
            'Remove and destroy mummified fruit, infected leaves, and dead canes.',
            'Apply fungicides (myclobutanil, mancozeb, or captan) from bud break through fruit set.',
            'Prune vines to improve air circulation and sunlight penetration.',
            'Keep the vineyard floor clean of fallen debris.',
            'Plant resistant grape varieties where available.',
        ]
    },
    'Grape___Esca_(Black_Measles)': {
        'cause': 'Caused by a complex of fungi including **Phaeomoniella chlamydospora**, **Phaeoacremonium spp.**, and **Fomitiporia spp.** They colonize the vascular system through pruning wounds.',
        'symptoms': 'Interveinal striping or "tiger stripe" pattern on leaves. Dark spots may appear on berries. Internal wood shows dark streaking and decay.',
        'solution': [
            'Protect pruning wounds with fungicide paste or wound sealant.',
            'Delay pruning until late in the dormant season to reduce infection risk.',
            'Remove and destroy severely infected vines (trunk renewal if possible).',
            'Apply preventive trunk treatments with biocontrol agents (Trichoderma spp.).',
            'There is no cure; focus on prevention and management of spread.',
        ]
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'cause': 'Caused by the fungus **Pseudocercospora vitis** (formerly Isariopsis). Thrives in warm, humid environments and spreads via wind and rain.',
        'symptoms': 'Dark brown spots with yellowish halos on leaves. Spots may enlarge and merge, causing premature leaf drop.',
        'solution': [
            'Apply fungicides (mancozeb or copper-based) at early signs of infection.',
            'Remove and destroy infected leaves and debris.',
            'Prune vines for better air circulation.',
            'Avoid overhead irrigation; water at the base of vines.',
            'Maintain vineyard hygiene by clearing weeds and fallen material.',
        ]
    },
    'Grape___healthy': {
        'cause': 'No disease detected.',
        'symptoms': 'The plant appears healthy with no visible signs of infection.',
        'solution': [
            'Continue regular monitoring for early signs of disease.',
            'Maintain proper canopy management with timely pruning.',
            'Apply preventive fungicide sprays during high-risk wet periods.',
        ]
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'cause': 'Caused by the bacterium **Candidatus Liberibacter asiaticus**, transmitted by the Asian citrus psyllid (*Diaphorina citri*). It is one of the most devastating citrus diseases worldwide.',
        'symptoms': 'Asymmetric blotchy mottling (yellowing) of leaves. Fruit remains small, lopsided, and green at the stem end. Bitter, off-flavor juice. Tree decline over several years.',
        'solution': [
            'Control psyllid vectors with insecticides (imidacloprid, spinosad) or biological control.',
            'Remove and destroy infected trees to prevent spread.',
            'Plant certified disease-free nursery stock.',
            'Use reflective mulch to deter psyllids.',
            'Monitor traps regularly for psyllid populations.',
            'There is currently no cure; prevention and vector control are critical.',
        ]
    },
    'Peach___Bacterial_spot': {
        'cause': 'Caused by the bacterium **Xanthomonas arboricola pv. pruni**. It spreads via rain splash and wind-driven rain, especially in warm, humid climates.',
        'symptoms': 'Small, dark, water-soaked spots on leaves that turn angular and may drop out ("shot-hole" appearance). Fruit develops sunken, cracked lesions.',
        'solution': [
            'Plant resistant peach varieties (e.g., Contender, Redhaven).',
            'Apply copper-based bactericides during dormant season and early spring.',
            'Avoid overhead irrigation to minimize leaf wetness.',
            'Prune to improve air circulation.',
            'Maintain balanced nitrogen fertilization (excess promotes susceptibility).',
        ]
    },
    'Peach___healthy': {
        'cause': 'No disease detected.',
        'symptoms': 'The plant appears healthy with no visible signs of infection.',
        'solution': [
            'Continue regular monitoring for early signs of disease.',
            'Apply dormant sprays in late winter for preventive care.',
            'Maintain proper pruning and fertilization.',
        ]
    },
    'Pepper,_bell___Bacterial_spot': {
        'cause': 'Caused by **Xanthomonas campestris pv. vesicatoria**. It spreads through contaminated seed, rain splash, and warm, humid conditions.',
        'symptoms': 'Small, dark, water-soaked spots on leaves that enlarge and become brown with yellow halos. Raised, scabby spots on fruit.',
        'solution': [
            'Use certified disease-free seeds and transplants.',
            'Apply copper-based sprays combined with mancozeb as a preventive.',
            'Practice crop rotation (avoid planting peppers/tomatoes in the same spot for 2–3 years).',
            'Remove and destroy infected plants immediately.',
            'Avoid working in the field when plants are wet to prevent spreading bacteria.',
        ]
    },
    'Pepper,_bell___healthy': {
        'cause': 'No disease detected.',
        'symptoms': 'The plant appears healthy with no visible signs of infection.',
        'solution': [
            'Continue regular monitoring for early signs of disease.',
            'Maintain proper watering and avoid wetting the foliage.',
            'Practice crop rotation and field sanitation.',
        ]
    },
    'Potato___Early_blight': {
        'cause': 'Caused by the fungus **Alternaria solani**. It thrives in warm, humid weather and survives in infected plant debris and soil.',
        'symptoms': 'Dark brown to black concentric ring lesions ("target spots") on older lower leaves. Leaves yellow and die from the bottom up.',
        'solution': [
            'Apply fungicides (chlorothalonil, mancozeb, or azoxystrobin) at first sign of disease.',
            'Practice crop rotation (avoid planting potatoes/tomatoes in the same field for 2+ years).',
            'Remove and destroy infected plant debris after harvest.',
            'Maintain adequate plant nutrition, especially nitrogen and potassium.',
            'Use certified disease-free seed potatoes.',
        ]
    },
    'Potato___Late_blight': {
        'cause': 'Caused by the oomycete **Phytophthora infestans**. This is the disease responsible for the Irish Potato Famine. It spreads rapidly in cool, wet weather.',
        'symptoms': 'Large, dark, water-soaked lesions on leaves and stems. White fuzzy mold growth on leaf undersides. Tubers develop firm, dark, rotted areas.',
        'solution': [
            'Apply fungicides (metalaxyl, chlorothalonil, or copper-based) preventively during wet weather.',
            'Plant resistant potato varieties.',
            'Destroy all volunteer potato plants and cull piles.',
            'Avoid overhead irrigation and water early in the day.',
            'Harvest tubers promptly and store in cool, dry conditions.',
            'Monitor weather forecasts—spray immediately when cool, wet conditions are predicted.',
        ]
    },
    'Potato___healthy': {
        'cause': 'No disease detected.',
        'symptoms': 'The plant appears healthy with no visible signs of infection.',
        'solution': [
            'Continue regular monitoring for early signs of disease.',
            'Maintain proper soil drainage and avoid overwatering.',
            'Use certified seed potatoes for planting.',
        ]
    },
    'Raspberry___healthy': {
        'cause': 'No disease detected.',
        'symptoms': 'The plant appears healthy with no visible signs of infection.',
        'solution': [
            'Continue regular monitoring for early signs of disease.',
            'Remove old fruiting canes after harvest.',
            'Maintain proper spacing for air circulation.',
        ]
    },
    'Soybean___healthy': {
        'cause': 'No disease detected.',
        'symptoms': 'The plant appears healthy with no visible signs of infection.',
        'solution': [
            'Continue regular monitoring for early signs of disease.',
            'Practice crop rotation and proper weed management.',
            'Test soil regularly and maintain proper nutrient levels.',
        ]
    },
    'Squash___Powdery_mildew': {
        'cause': 'Caused by the fungi **Podosphaera xanthii** or **Erysiphe cichoracearum**. Thrives in warm, dry weather with high humidity and poor air circulation.',
        'symptoms': 'White, powdery fungal growth on upper and lower leaf surfaces. Leaves yellow, wither, and die. Reduced fruit quality and yield.',
        'solution': [
            'Apply fungicides (sulfur, potassium bicarbonate, or neem oil) at first sign of infection.',
            'Plant resistant squash varieties.',
            'Space plants adequately for good air circulation.',
            'Avoid overhead watering; water at the soil level.',
            'Remove and destroy severely infected leaves.',
        ]
    },
    'Strawberry___Leaf_scorch': {
        'cause': 'Caused by the fungus **Diplocarpon earlianum**. It spreads through rain splash and overhead irrigation in warm, humid weather.',
        'symptoms': 'Irregular dark purple to brown spots on leaves. Spots may merge, causing leaves to look "scorched" or burnt. Severe infection reduces fruit yield.',
        'solution': [
            'Apply fungicides (captan or myclobutanil) from bloom onward.',
            'Remove and destroy infected leaves and runners.',
            'Plant resistant strawberry varieties.',
            'Use drip irrigation instead of overhead watering.',
            'Renovate strawberry beds annually to remove old, infected foliage.',
        ]
    },
    'Strawberry___healthy': {
        'cause': 'No disease detected.',
        'symptoms': 'The plant appears healthy with no visible signs of infection.',
        'solution': [
            'Continue regular monitoring for early signs of disease.',
            'Mulch around plants to reduce soil splash.',
            'Remove runners and old leaves to maintain plant vigor.',
        ]
    },
    'Tomato___Bacterial_spot': {
        'cause': 'Caused by **Xanthomonas spp.** bacteria. It spreads through contaminated seeds, transplants, rain splash, and warm, humid conditions.',
        'symptoms': 'Small, dark, water-soaked spots on leaves, stems, and fruit. Leaf spots may have yellow halos. Fruit spots are raised and scabby.',
        'solution': [
            'Use certified disease-free seeds and transplants.',
            'Apply copper-based bactericides as a preventive spray.',
            'Practice crop rotation (avoid tomatoes/peppers in the same area for 2–3 years).',
            'Remove and destroy infected plants promptly.',
            'Avoid working with plants when wet to prevent bacterial spread.',
        ]
    },
    'Tomato___Early_blight': {
        'cause': 'Caused by the fungus **Alternaria solani**. Survives in soil and infected plant debris. Favored by warm temperatures and wet conditions.',
        'symptoms': 'Dark brown circular spots with concentric rings ("target spots") on lower, older leaves. Leaves yellow and drop. Stem and fruit lesions may also occur.',
        'solution': [
            'Apply fungicides (chlorothalonil, mancozeb, or copper-based) at first symptom.',
            'Mulch around plants to prevent soil splashing onto leaves.',
            'Practice crop rotation with non-solanaceous crops.',
            'Stake or cage plants to keep foliage off the ground.',
            'Water at the base of plants; avoid overhead irrigation.',
        ]
    },
    'Tomato___Late_blight': {
        'cause': 'Caused by the oomycete **Phytophthora infestans**. Spreads extremely rapidly in cool, wet weather via wind-blown spores.',
        'symptoms': 'Large, irregularly shaped, water-soaked dark lesions on leaves and stems. White fuzzy mold on leaf undersides. Fruit develops firm, dark, greasy-looking rot.',
        'solution': [
            'Apply fungicides (metalaxyl, chlorothalonil, or copper-based) preventively in wet weather.',
            'Remove and destroy all infected plant material immediately.',
            'Do not compost infected plants.',
            'Plant resistant tomato varieties.',
            'Improve air circulation by proper spacing and staking.',
            'Monitor weather—act immediately when cool, wet conditions are forecast.',
        ]
    },
    'Tomato___Leaf_Mold': {
        'cause': 'Caused by the fungus **Passalora fulva** (formerly Cladosporium fulvum). Thrives in high humidity and poor ventilation, especially in greenhouses.',
        'symptoms': 'Pale green-to-yellow spots on upper leaf surfaces. Olive-green to grayish-brown velvety mold on leaf undersides. Leaves may curl and wither.',
        'solution': [
            'Improve ventilation and air circulation in greenhouses.',
            'Reduce humidity by watering at the base and avoiding overhead irrigation.',
            'Apply fungicides (chlorothalonil or copper-based) as needed.',
            'Plant resistant tomato varieties.',
            'Remove and destroy infected leaves promptly.',
        ]
    },
    'Tomato___Septoria_leaf_spot': {
        'cause': 'Caused by the fungus **Septoria lycopersici**. Survives in infected plant debris and spreads via rain splash in warm, wet conditions.',
        'symptoms': 'Many small, circular spots (1–3 mm) with dark borders and grayish-white centers with tiny black dots (pycnidia). Starts on lower leaves and progresses upward.',
        'solution': [
            'Apply fungicides (chlorothalonil, mancozeb, or copper-based) at first sign of disease.',
            'Remove and destroy lower infected leaves as soon as spots appear.',
            'Mulch around plants to prevent rain splashing soil onto leaves.',
            'Practice crop rotation (avoid tomatoes in the same spot for 2+ years).',
            'Stake plants to improve air circulation.',
        ]
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'cause': 'Caused by the **Two-Spotted Spider Mite** (*Tetranychus urticae*), a tiny arachnid pest (not a disease). Thrives in hot, dry, dusty conditions.',
        'symptoms': 'Tiny yellow or white stippling (dots) on upper leaf surfaces. Fine webbing on leaf undersides. Leaves bronze, dry up, and drop in severe infestations.',
        'solution': [
            'Spray plants with a strong jet of water to dislodge mites.',
            'Apply miticides (abamectin) or insecticidal soap / neem oil.',
            'Introduce natural predators such as ladybugs or predatory mites (*Phytoseiulus persimilis*).',
            'Increase humidity around plants (mites prefer dry conditions).',
            'Avoid broad-spectrum insecticides that kill natural predators.',
            'Remove heavily infested leaves and destroy them.',
        ]
    },
    'Tomato___Target_Spot': {
        'cause': 'Caused by the fungus **Corynespora cassiicola**. Thrives in warm, humid environments and spreads through wind and rain splash.',
        'symptoms': 'Brown circular spots with concentric rings on leaves, stems, and fruit. Spots may enlarge and cause defoliation.',
        'solution': [
            'Apply fungicides (chlorothalonil, azoxystrobin, or mancozeb) preventively.',
            'Remove lower leaves to improve air circulation near the soil.',
            'Practice crop rotation with non-host crops.',
            'Avoid overhead irrigation.',
            'Remove and destroy crop debris after harvest.',
        ]
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'cause': 'Caused by the **Tomato Yellow Leaf Curl Virus (TYLCV)**, transmitted by the whitefly *Bemisia tabaci*. It is a devastating viral disease with no cure.',
        'symptoms': 'Upward curling and cupping of leaves with yellow margins. Stunted growth, small leaves, and flower drop. Severely reduced fruit production.',
        'solution': [
            'Control whitefly populations with insecticides (imidacloprid, pyriproxyfen) or sticky yellow traps.',
            'Use insect-proof netting or row covers to exclude whiteflies.',
            'Plant TYLCV-resistant tomato varieties.',
            'Remove and destroy infected plants immediately to stop virus spread.',
            'Introduce natural whitefly predators (Encarsia formosa, ladybugs).',
            'Avoid planting tomatoes near other infected crops.',
        ]
    },
    'Tomato___Tomato_mosaic_virus': {
        'cause': 'Caused by the **Tomato Mosaic Virus (ToMV)**. It spreads through contaminated seeds, tools, hands, and plant-to-plant contact. Extremely stable and can persist on surfaces for years.',
        'symptoms': 'Mosaic pattern of light and dark green on leaves. Leaf distortion, curling, and fern-like appearance. Fruit may show internal browning or uneven ripening.',
        'solution': [
            'Use certified virus-free seeds and transplants.',
            'Disinfect all tools, stakes, and hands with a 10% bleach solution or milk solution before handling plants.',
            'Remove and destroy infected plants immediately.',
            'Do not use tobacco products near plants (tobacco mosaic virus is related and can cross-infect).',
            'Plant resistant tomato varieties with Tm-2 or Tm-2² resistance genes.',
            'Avoid touching healthy plants after handling infected ones.',
        ]
    },
    'Tomato___healthy': {
        'cause': 'No disease detected.',
        'symptoms': 'The plant appears healthy with no visible signs of infection.',
        'solution': [
            'Continue regular monitoring for early signs of disease.',
            'Maintain proper watering, staking, and pruning.',
            'Practice crop rotation and field sanitation.',
        ]
    },
}


# Sidebar
st.sidebar.title("Dashboard")

app_mode = st.sidebar.selectbox(
    "Select Page",
    ["Home", "About", "Disease Recognition"]
)


# Home Page
if app_mode == "Home":

    st.header("PLANT DISEASE RECOGNITION SYSTEM")

    image_path = "home_page.jpeg"
    st.image(image_path, width="stretch")

    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help identify plant diseases efficiently.

    ### How It Works

    1. **Upload Image:** Go to the **Disease Recognition** page.
    2. **Analysis:** The model processes the uploaded leaf image.
    3. **Results:** The predicted disease is displayed.

    ### Why Choose Us?

    - **Accuracy:** Deep learning-based disease classification.
    - **User-Friendly:** Simple interface.
    - **Fast:** Predictions are generated within seconds.

    ### Get Started

    Select **Disease Recognition** from the sidebar.
    """)


# About Page
elif app_mode == "About":

    st.header("About")

    st.markdown("""
    #### About Dataset

    This dataset consists of approximately 87K RGB images
    of healthy and diseased crop leaves categorized into
    38 different classes.

    The dataset is divided into training and validation sets.

    #### Content

    1. Train: 70,295 images
    2. Test: 33 images
    3. Validation: 17,572 images
    """)


# Prediction Page
elif app_mode == "Disease Recognition":

    st.header("Disease Recognition")

    test_image = st.file_uploader(
        "Choose an Image:",
        type=["jpg", "jpeg", "png"]
    )

    if test_image is not None:

        if st.button("Predict"):

            st.snow()

            # Check if the image looks like a plant/leaf
            test_image.seek(0)  # Reset file pointer
            plant_check = is_likely_plant_image(test_image)
            test_image.seek(0)  # Reset file pointer again for model

            result_index, confidence = model_prediction(test_image)
            test_image.seek(0)  # Reset for displaying

            # Confidence threshold — if the model is unsure, the image
            # is likely not a valid plant leaf
            CONFIDENCE_THRESHOLD = 0.50

            if confidence < CONFIDENCE_THRESHOLD or not plant_check:
                # Irrelevant image detected — warn the user
                st.image(
                    test_image,
                    caption="Uploaded Image",
                    width="stretch"
                )
                st.warning(
                    "⚠️ **This doesn't look like a valid plant leaf image!**\n\n"
                    "Please upload a clear image of a **plant leaf** for accurate disease detection.\n\n"
                    "**Tips for best results:**\n"
                    "- Upload a close-up photo of a single leaf\n"
                    "- Make sure the leaf is clearly visible and well-lit\n"
                    "- Avoid uploading images of non-plant objects (cars, people, animals, etc.)\n\n"
                    f"*Model confidence: {confidence * 100:.1f}%*"
                )
            else:
                # Valid plant image — proceed with prediction
                # Parse the predicted class name
                predicted_class = class_name[result_index]
                parts = predicted_class.split("___")
                plant_name = parts[0].replace("_", " ")
                condition = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"

                st.write("### Our Prediction")

                if "healthy" in predicted_class.lower():
                    # Healthy plant — show image and success message
                    st.image(
                        test_image,
                        caption="Uploaded Image",
                        width="stretch"
                    )
                    st.success(
                        f"✅ The **{plant_name}** plant looks **Healthy**! "
                        "No disease detected."
                    )
                    # Show care tips
                    info = disease_info.get(predicted_class, None)
                    if info:
                        st.info("💡 **Care Tips:**")
                        for tip in info['solution']:
                            st.markdown(f"- {tip}")
                else:
                    # Diseased plant — two-column layout
                    col_left, col_right = st.columns([1, 1])

                    with col_left:
                        st.image(
                            test_image,
                            caption="Uploaded Image",
                            width="stretch"
                        )

                    with col_right:
                        st.error(
                            f"⚠️ **Disease Detected!**\n\n"
                            f"**Plant:** {plant_name}\n\n"
                            f"**Disease:** {condition}"
                        )

                        info = disease_info.get(predicted_class, None)
                        if info:
                            st.markdown("---")
                            st.markdown(f"🦠 **Cause:**\n\n{info['cause']}")
                            st.markdown(f"🔍 **Symptoms:**\n\n{info['symptoms']}")
                            st.markdown("💊 **Recommended Solutions:**")
                            for i, sol in enumerate(info['solution'], 1):
                                st.markdown(f"{i}. {sol}")
    else:
        st.info("📤 Please upload a plant leaf image to get started.")