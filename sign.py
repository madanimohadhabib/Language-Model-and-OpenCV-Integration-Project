import cv2
import mediapipe as mp
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Initialisation des modules MediaPipe pour la détection des mains
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Chargement du modèle GPT et du tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_response(sign_language_text):
    inputs = tokenizer.encode(sign_language_text, return_tensors='pt')
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)  # Création du masque d'attention
    outputs = model.generate(inputs, attention_mask=attention_mask, max_length=50, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Fonction pour simuler la reconnaissance des signes (à remplacer par une vraie implémentation)
def recognize_sign(hand_landmarks):
    # Ici, vous devez implémenter la logique pour convertir les landmarks en texte de langue des signes
    # Cette fonction est juste un exemple et renvoie un texte fixe
    return "Bonjour, comment allez-vous?"

# Capture vidéo
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Conversion de l'image en RGB pour MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    # Dessiner les points de détection des mains
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Reconnaître le signe et obtenir une réponse
            sign_language_text = recognize_sign(hand_landmarks)
            response = generate_response(sign_language_text)
            print("GPT Response:", response)
    
    # Affichage de la vidéo avec les annotations
    cv2.imshow("Langue des Signes", img)
    
    # Quitter la boucle en appuyant sur la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
