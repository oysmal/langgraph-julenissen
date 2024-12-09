# Julenissen med LangGraph

For å kjøre koden:

Sørg for å ha Python 3.13.0 installert.

1. Installer [PyEnv](https://github.com/pyenv/pyenv) og [PyEnv-virtualenv](https://github.com/pyenv/pyenv-virtualenv)
2. Lag et nytt virtual environment med `pyenv virtualenv 3.13.0 langgraph-julenissen`
3. Aktiver virtual environmentet med `pyenv activate langgraph-julenissen`
4. Installer avhengigheter med `pip install -r requirements.txt`
5. Kjør `python test.py` for å kjøre den ferdige koden fra julekalender-luken. Sørg for å ha DB_URI og OPENAI_API_KEY satt i environment-variabler.
6. Kjør `streamlit run main.py` for å kjøre streamlit-applikasjonen. Du mnå også kopiere `secrets.toml.example` til `./.streamlit/secrets.toml`, og fylle ut med dine verdier.
