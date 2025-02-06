import streamlit as st
import pandas as pd
import altair as alt

# Titel und Beschreibung der App
st.title("SCORING Analyse für die Qualitative Investitionsrechnung")
st.write("Geben Sie für jedes Kriterium eine Bewertung ein (Skala 1 bis 10, wobei 10 die beste Bewertung ist).")

# Feste Kriterien und ihre Gewichtungen (in Prozentpunkten, insgesamt 100)
criteria = {
    "Marktpotenzial": 30,
    "Innovationsgrad": 25,
    "Wettbewerbsfähigkeit": 20,
    "Strategische Bedeutung": 15,
    "Nachhaltigkeit": 10
}

# Auswahl der Anzahl Investitionen (1 bis 3)
num_investments = st.number_input(
    "Anzahl der zu vergleichenden Investitionen", 
    min_value=1, 
    max_value=3, 
    value=1, 
    step=1
)

# Dictionary zum Speichern der Investitionsdaten
investments = {}

# Für jede Investition werden Name und Kriterien-Bewertungen eingegeben
for i in range(1, num_investments + 1):
    st.subheader(f"Investition {i}")
    
    # Möglichkeit, einen eigenen Namen festzulegen
    name = st.text_input(f"Name der Investition {i}", value=f"Investition {i}", key=f"name_{i}")
    
    # Bewertungen für jedes Kriterium
    ratings = {}
    for crit in criteria:
        ratings[crit] = st.slider(
            f"Bewertung für **{crit}**", 
            min_value=1, 
            max_value=10, 
            value=5, 
            key=f"{crit}_{i}"
        )
    investments[name] = ratings

# Berechnung der gewichteten Scores und des Gesamt-Scores für jede Investition
results = []
for inv_name, rating_dict in investments.items():
    # Berechnung des gewichteten Scores pro Kriterium: Bewertung * Gewichtung
    weighted_scores = {crit: rating_dict[crit] * criteria[crit] for crit in criteria}
    total_weighted = sum(weighted_scores.values())
    total_weights = sum(criteria.values())
    overall_score = total_weighted / total_weights

    # Aufbau eines Ergebniseintrags (hier werden auch die Einzelbewertungen angezeigt)
    result = {"Investition": inv_name, "Gesamt-Score": round(overall_score, 2)}
    for crit in criteria:
        result[crit] = rating_dict[crit]
    results.append(result)

# Ergebnisse in einer DataFrame zusammenfassen
df = pd.DataFrame(results)

st.subheader("Vergleichstabelle der Investitionen")
st.table(df)

# Balkendiagramm zum Vergleich der Gesamt-Scores
chart = alt.Chart(df).mark_bar().encode(
    x=alt.X("Investition", sort=None),
    y=alt.Y("Gesamt-Score", title="Gesamt-Score"),
    color=alt.Color("Investition", legend=None),
    tooltip=["Investition", "Gesamt-Score"]
).properties(
    width=600,
    height=400,
    title="Vergleich der Gesamt-Scores"
)

st.altair_chart(chart, use_container_width=True)

# Empfehlung: Welche Investition hat den höchsten Score?
if not df.empty:
    best_investment = df.loc[df["Gesamt-Score"].idxmax()]
    st.markdown(f"### Empfohlene Investition:")
    st.markdown(f"**{best_investment['Investition']}** mit einem Gesamt-Score von **{best_investment['Gesamt-Score']}**")