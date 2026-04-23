import streamlit as st
import httpx

# Configuration de la page
st.set_page_config(page_title="Météo Agent Mistral", page_icon="🌤️")
st.title("🌤️ Assistant Météo Mistral")
st.markdown("Posez vos questions sur la météo (ex: Quel temps fait-il à Paris ?)")

# Initialisation de l'historique dans la session Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage des messages de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie utilisateur
if prompt := st.chat_input("Demandez la météo..."):
    # Ajouter le message utilisateur à l'interface
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Appel à l'API FastAPI
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("*L'assistant réfléchit...*")
        
        try:
            # On envoie le message et l'historique à FastAPI
            payload = {
                "message": prompt,
                "history": st.session_state.messages[:-1] # On exclut le dernier message pour l'historique
            }
            
            response = httpx.post("http://127.0.0.1:8000/chat", json=payload, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                full_response = data["response"]
                tool_info = data.get("tool_used")
                
                # Affichage si un outil a été utilisé
                if tool_info:
                    st.caption(f"🔧 Outil utilisé : `{tool_info}`")
                
                message_placeholder.markdown(full_response)
                
                # Sauvegarde de la réponse de l'assistant
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                st.error(f"Erreur API : {response.status_code}")
                
        except Exception as e:
            st.error(f"Erreur de connexion : {e}")

# Bouton pour effacer le chat
if st.button("Effacer l'historique"):
    st.session_state.messages = []
    st.rerun()