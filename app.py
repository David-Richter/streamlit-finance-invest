import streamlit as st
import pandas as pd

###############################################################################
# Finanzmathematische Funktionen
###############################################################################
def future_value(pv: float, i: float, n: int) -> float:
    """
    Endwert (EW_n) einer heutigen Zahlung (Barwert BW_0) nach n Jahren.
    Formel: EW_n = pv * (1 + i)^n
    """
    return pv * (1 + i)**n

def present_value(fv: float, i: float, n: int) -> float:
    """
    Barwert (BW_0) einer zukünftigen Zahlung (EW_n) in n Jahren.
    Formel: BW_0 = fv / (1 + i)^n
    """
    return fv / ((1 + i)**n)

def annuity_future_value(b: float, i: float, n: int) -> float:
    """
    Endwert einer Rente (nachschüssig) mit Rate b pro Jahr, Zinssatz i, Laufzeit n.
    Formel: EW_n = b * ((1+i)^n - 1) / i
    """
    q = 1 + i
    return b * ((q**n - 1) / i)

def annuity_present_value(b: float, i: float, n: int) -> float:
    """
    Barwert einer Rente (nachschüssig) mit Rate b pro Jahr, Zinssatz i, Laufzeit n.
    Formel: BW_0 = b * ( (1+i)^n - 1 ) / ( i * (1+i)^n )
    """
    q = 1 + i
    return b * ((q**n - 1) / (i * q**n))

def annuity_from_present_value(pv: float, i: float, n: int) -> float:
    """
    Gesuchte Rate b, wenn der Barwert einer Rente (BW_0) gegeben ist.
    b = BW_0 * ( i * (1+i)^n ) / ( (1+i)^n - 1 )
    """
    q = 1 + i
    return pv * (i * q**n) / (q**n - 1)

def annuity_from_future_value(fv: float, i: float, n: int) -> float:
    """
    Gesuchte Rate b, wenn der Endwert einer Rente (EW_n) gegeben ist.
    b = EW_n * ( i / ( (1+i)^n - 1 ) )
    """
    q = 1 + i
    return fv * (i / (q**n - 1))

def npv_of_cashflow_series(cashflows: list[float], i: float) -> float:
    """
    NPV (Nettobarwert) einer CF‐Reihe c_1,...,c_n, diskontiert mit i.
    NPV = ∑( c_t / (1 + i)^t ), t=1..n
    """
    npv = 0.0
    for t, cf in enumerate(cashflows, start=1):
        npv += cf / ((1 + i)**t)
    return npv

def fv_of_cashflow_series(cashflows: list[float], i: float) -> float:
    """
    Endwert (EW_n) einer allgemeinen CF‐Reihe c_1,...,c_n zum Zeitpunkt n.
    EW_n = ∑( c_t * (1 + i)^(n - t) ), t=1..n
    """
    n = len(cashflows)
    fv = 0.0
    for t, cf in enumerate(cashflows, start=1):
        fv += cf * ((1 + i)**(n - t))
    return fv


###############################################################################
# Hilfsfunktionen für Visualisierungen
###############################################################################
def visualize_single_payment_growth(pv: float, i: float, n: int):
    """
    Erzeugt eine DataFrame für den jährlich wachsenden Wert einer Einmalzahlung pv
    bei Zinssatz i bis Jahr n (d.h. Jahr 0 bis n).
    """
    values = []
    for year in range(n+1):
        val = pv * (1 + i)**year
        values.append(val)
    df = pd.DataFrame({
        "Jahr": list(range(n+1)),
        "Wert": values
    })
    df.set_index("Jahr", inplace=True)
    return df

def visualize_annuity_future(b: float, i: float, n: int):
    """
    Zeigt, wie sich eine jährliche Rente b bis zum Endwert entwickelt.
    Jedes Jahr kommt eine Zahlung b dazu, plus Zins auf bereits angefallene Beträge.
    """
    q = 1 + i
    balance = 0.0
    timeline = []
    for year in range(1, n+1):
        # Vor Rentenzahlung zinsen wir das Vorjahr auf
        balance *= q
        # Dann fließt die Rente
        balance += b
        timeline.append(balance)
    df = pd.DataFrame({"Jahr": list(range(1, n+1)), "Saldo": timeline})
    df.set_index("Jahr", inplace=True)
    return df

def visualize_annuity_present(b: float, i: float, n: int):
    """
    Zeigt kumulativ den diskontierten Betrag jeder Rentenzahlung,
    um den Barwert zu illustrieren.
    """
    q = 1 + i
    timeline = []
    cumulative = 0.0
    for t in range(1, n+1):
        discounted = b / (q**t)
        cumulative += discounted
        timeline.append(cumulative)
    df = pd.DataFrame({"Jahr": list(range(1, n+1)), "kumul. Barwert": timeline})
    df.set_index("Jahr", inplace=True)
    return df

def visualize_cashflow_series(cashflows: list[float]):
    """
    Stellt die einzelnen Zahlungen pro Periode als Balken dar.
    """
    df = pd.DataFrame({
        "Jahr": list(range(1, len(cashflows)+1)),
        "Cashflow": cashflows
    })
    df.set_index("Jahr", inplace=True)
    return df


###############################################################################
# Streamlit App
###############################################################################

st.title("Finanzmathematische Berechnungen mit Visualisierung")

st.markdown("""
Willkommen zu dieser **App**, die finanzmathematische Grundlagen
(z. B. Barwert, Endwert, Rente, NPV) berechnet **und** zur Veranschaulichung
eine kleine **grafische Darstellung** erzeugt.  
""")

calc_choice = st.sidebar.radio(
    "Bitte wähle eine Berechnungsart:",
    (
        "Einmalige Zahlung (Barwert / Endwert)",
        "Renten-Berechnungen (Bar- und Endwert)",
        "Rentenhöhe aus Barwert / Endwert",
        "Zahlungsreihe (NPV / FV)"
    )
)


###############################################################################
# 1. Einmalige Zahlung
###############################################################################
if calc_choice == "Einmalige Zahlung (Barwert / Endwert)":
    st.subheader("Einmalige Zahlung: Barwert und Endwert")
    tab = st.radio("Berechnung auswählen", ("Endwert", "Barwert"), horizontal=True)

    if tab == "Endwert":
        pv_val = st.number_input("Aktueller Betrag (BW₀)", value=1000.0, step=100.0)
        i_val = st.number_input("Zinssatz i (Dezimal, z.B. 0.06 = 6%)", value=0.06)
        n_val = st.number_input("Anzahl Perioden (n)", value=8, step=1)

        if st.button("Berechne Endwert"):
            result = future_value(pv_val, i_val, n_val)
            st.success(f"Endwert (EWₙ) = {result:,.2f} €")

            # Visualisierung: Wachstum pro Jahr
            df_chart = visualize_single_payment_growth(pv_val, i_val, n_val)
            st.line_chart(df_chart)

    else:  # Barwert
        fv_val = st.number_input("Zukünftiger Betrag (EWₙ)", value=1000.0, step=100.0)
        i_val = st.number_input("Zinssatz i (Dezimal, z.B. 0.06 = 6%)", value=0.06)
        n_val = st.number_input("Anzahl Perioden (n)", value=8, step=1)

        if st.button("Berechne Barwert"):
            result = present_value(fv_val, i_val, n_val)
            st.success(f"Barwert (BW₀) = {result:,.2f} €")

            # Visualisierung: Wir können ebenfalls den "Wachstumsverlauf" rückwärts zeigen
            # => Hier aber optional. Wer mag, kann die Discounting-Kurve darstellen.
            st.write("**Optionale Visualisierung**: Wenn du die *Aufzinsung* (EW) bei jeder Periode betrachtest, "
                     "wäre es der gleiche Graph wie bei 'Endwert', nur rückwärts interpretiert.")
            # Kein zusätzliches Diagramm hier, um es nicht zu kompliziert zu machen.


###############################################################################
# 2. Renten-Berechnungen
###############################################################################
elif calc_choice == "Renten-Berechnungen (Bar- und Endwert)":
    st.subheader("Renten-Berechnungen (gleichbleibende Zahlungen b)")
    tab = st.radio("Berechnung auswählen", ("Rentenbarwert", "Rentenendwert"), horizontal=True)

    if tab == "Rentenbarwert":
        b_val = st.number_input("Jährliche Rentenzahlung b", value=1000.0, step=100.0)
        i_val = st.number_input("Zinssatz i", value=0.06)
        n_val = st.number_input("Anzahl Perioden n", value=8, step=1)

        if st.button("Berechne Renten-Barwert"):
            result = annuity_present_value(b_val, i_val, n_val)
            st.success(f"Barwert der Rente = {result:,.2f} €")
            
            # Visualisierung: kumul. diskontierter Wert
            df_barwert = visualize_annuity_present(b_val, i_val, n_val)
            st.line_chart(df_barwert)

    else:  # Rentenendwert
        b_val = st.number_input("Jährliche Rentenzahlung b", value=1000.0, step=100.0)
        i_val = st.number_input("Zinssatz i", value=0.06)
        n_val = st.number_input("Anzahl Perioden n", value=8, step=1)

        if st.button("Berechne Renten-Endwert"):
            result = annuity_future_value(b_val, i_val, n_val)
            st.success(f"Endwert der Rente = {result:,.2f} €")

            # Visualisierung: Wachstum der Rente je Periode
            df_endwert = visualize_annuity_future(b_val, i_val, n_val)
            st.line_chart(df_endwert)


###############################################################################
# 3. Rentenhöhe aus Barwert / Endwert
###############################################################################
elif calc_choice == "Rentenhöhe aus Barwert / Endwert":
    st.subheader("Gesuchte Rentenhöhe b aus gegebenem Barwert oder Endwert")
    tab = st.radio("Was ist vorgegeben?", ("Barwert (BW₀)", "Endwert (EWₙ)"), horizontal=True)

    if tab == "Barwert (BW₀)":
        pv_val = st.number_input("Barwert (BW₀)", value=6209.79, step=100.0)
        i_val = st.number_input("Zinssatz i", value=0.06)
        n_val = st.number_input("Anzahl Perioden n", value=8, step=1)

        if st.button("Berechne b aus Barwert"):
            b_stern = annuity_from_present_value(pv_val, i_val, n_val)
            st.success(f"Rentenhöhe b = {b_stern:,.2f} €")
            
            # Hier könntest du optional eine Visualisierung ergänzen, z.B. 
            # wie sich eine Rente b zusammensetzt, wenn der Barwert gegeben ist.

    else:  # Endwert (EWₙ)
        fv_val = st.number_input("Endwert (EWₙ)", value=9897.47, step=100.0)
        i_val = st.number_input("Zinssatz i", value=0.06)
        n_val = st.number_input("Anzahl Perioden n", value=8, step=1)

        if st.button("Berechne b aus Endwert"):
            b_stern = annuity_from_future_value(fv_val, i_val, n_val)
            st.success(f"Rentenhöhe b = {b_stern:,.2f} €")
            
            # Auch hier könnte man eine Visualisierung anfügen, ähnlich wie 
            # in 'visualize_annuity_future', nur den Wert b dynamisch einsetzen.


###############################################################################
# 4. Zahlungsreihe (NPV / FV)
###############################################################################
else:  # "Zahlungsreihe (NPV / FV)"
    st.subheader("Zahlungsreihe (unregelmäßige Zahlungen) – NPV / FV")
    st.markdown("""
    Gib hier eine beliebige Reihe an jährlichen **Cashflows** ein.
    Dann kannst du den **Barwert (NPV)** oder **Endwert (FV)** berechnen
    und ein Diagramm anzeigen lassen.
    """)

    cf_str = st.text_input("Cashflows als Komma-getrennte Liste (z.B. '100, 200, 300'):", "100,200,300")
    i_val = st.number_input("Zinssatz i", value=0.06)
    tab = st.radio("Was möchtest du berechnen?", ("Barwert (NPV)", "Endwert (FV)"), horizontal=True)

    if st.button("Berechnen"):
        try:
            cashflows = [float(x.strip()) for x in cf_str.split(",")]
        except ValueError:
            st.error("Fehler beim Einlesen der Cashflows. Bitte gültige Zahlen im CSV-Format eingeben.")
            st.stop()

        if tab == "Barwert (NPV)":
            result = npv_of_cashflow_series(cashflows, i_val)
            st.success(f"Barwert (NPV) = {result:,.2f} €")

            # Visualisierung: Balkendiagramm der eingegebenen Cashflows
            df_cf = visualize_cashflow_series(cashflows)
            st.bar_chart(df_cf)

        else:
            result = fv_of_cashflow_series(cashflows, i_val)
            st.success(f"Endwert (FV) = {result:,.2f} €")

            # Visualisierung: Balkendiagramm der Cashflows
            df_cf = visualize_cashflow_series(cashflows)
            st.bar_chart(df_cf)


st.markdown("---")
st.caption("© 2025 – Workaround, weil ich keine Lust habe Brüche in einen Taschenrechner zu tippen... :D")