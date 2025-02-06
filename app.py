import streamlit as st
import pandas as pd
import altair as alt
import math
import sympy

###############################################################################
# Finanzmathematische Funktionen (numerische Kernfunktionen)
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
    Formel: BW_0 = b * ((1+i)^n - 1) / (i * (1+i)^n)
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

def payback_period(a0: float, annual_return: float) -> float:
    """
    Berechnet die statische Amortisationsdauer (AD) nach der Formel:
        AD = a0 / annual_return
    """
    if annual_return == 0:
        return float("inf")
    return a0 / annual_return

###############################################################################
# Visualisierungen
###############################################################################
def plot_payback(a0: float, annual_return: float):
    ad = payback_period(a0, annual_return)
    max_year = math.ceil(ad) + 1
    
    data = []
    for t in range(max_year + 1):
        cum_return = annual_return * t
        data.append({"Jahr": t, "Kumulierte Rückflüsse": cum_return})

    df = pd.DataFrame(data)
    base_chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("Jahr:Q"),
        y=alt.Y("Kumulierte Rückflüsse:Q"),
        tooltip=["Jahr", "Kumulierte Rückflüsse"]
    )
    rule = alt.Chart(pd.DataFrame({"y": [a0]})).mark_rule(color="red").encode(y="y")

    return (base_chart + rule).interactive()

def visualize_single_payment_growth(pv: float, i: float, n: int):
    values = []
    for year in range(n+1):
        val = pv * (1 + i)**year
        values.append(val)
    df = pd.DataFrame({
        "Jahr": list(range(n+1)),
        "Wert": values
    }).set_index("Jahr")
    return df

def visualize_annuity_future(b: float, i: float, n: int):
    q = 1 + i
    balance = 0.0
    timeline = []
    for year in range(1, n+1):
        balance *= q
        balance += b
        timeline.append(balance)
    df = pd.DataFrame({"Jahr": list(range(1, n+1)), "Saldo": timeline}).set_index("Jahr")
    return df

def visualize_annuity_present(b: float, i: float, n: int):
    q = 1 + i
    timeline = []
    cumulative = 0.0
    for t in range(1, n+1):
        discounted = b / (q**t)
        cumulative += discounted
        timeline.append(cumulative)
    df = pd.DataFrame({"Jahr": list(range(1, n+1)), "kumul. Barwert": timeline}).set_index("Jahr")
    return df

def visualize_cashflow_series(cashflows: list[float]):
    df = pd.DataFrame({
        "Jahr": list(range(1, len(cashflows)+1)),
        "Cashflow": cashflows
    }).set_index("Jahr")
    return df

###############################################################################
# Erklärung der Variablennamen (für LaTeX-Ausgabe)
###############################################################################
symbol_explanations = {
    "BW_0": "Barwert (heutiger Gegenwert) in Euro",
    "EW_n": "Endwert (künftiger Wert) in Euro",
    "b":    "Jährliche Rente (konstante Zahlung in Euro)",
    "i":    "Zinssatz pro Periode (dezimal, z. B. 0.06 = 6%)",
    "n":    "Anzahl der Perioden (z. B. Jahre)",
    "a_0":  "Anfangsinvestition (Anschaffungsausgabe) in Euro",
    "C":    "Jährlicher Rückfluss in Euro",
    "AD":   "Amortisationsdauer (Jahre)"
}

###############################################################################
# Hilfsfunktion: Zeige Sympy-Formel mitsamt Variablen-Erklärung
###############################################################################
def show_sympy_formula(
    expr: sympy.Expr,
    symbols_map: dict,
    result_label="Ergebnis",
    numeric=None
):
    """
    Zeigt die allgemeine Formel, die Erklärung der Variablen,
    die eingesetzten Werte und das numerische Ergebnis per LaTeX in Streamlit an.

    :param expr: Sympy expression
    :param symbols_map: Dict {Symbol: Wert, ...} mit den konkreten Einsetzungen
    :param result_label: z.B. 'BW_0'
    :param numeric: fertiges numerisches Ergebnis (float) oder None
    """
    from sympy import latex

    # 1) Erklärung der Variablen
    st.markdown("**Erklärung der verwendeten Variablen**:")
    # Sammle alle Symbole, die in expr oder in symbols_map auftauchen
    all_syms = set(expr.free_symbols) | set(symbols_map.keys())
    # all_syms ist eine Menge aus z.B. iSym, nSym, ...
    # -> nutze .name, um im dictionary nachzusehen
    for s in sorted(all_syms, key=lambda x: x.name):
        s_name = s.name if isinstance(s, sympy.Symbol) else str(s)
        if s_name in symbol_explanations:
            st.write(f"- **{s_name}**: {symbol_explanations[s_name]}")

    # 2) Allgemeine symbolische Formel
    st.markdown("**Allgemeine symbolische Formel**:")
    st.latex(f"{result_label} = {latex(expr)}")
    
    # 3) Eingesetzte tatsächliche Werte in der Formel
    st.markdown("**Eingesetzte Werte in der Formel**:")
    substituted_expr = expr.subs(symbols_map)
    st.latex(f"{result_label} = {latex(substituted_expr)}")

    # 4) Numerisches Ergebnis (sofern übergeben)
    if numeric is not None:
        st.markdown("**Numerisches Ergebnis**:")
        st.latex(f"{result_label} \\approx {numeric:,.2f}")

###############################################################################
# Streamlit-App
###############################################################################
st.title("Finanzmathematische Berechnungen mit Sympy-Formel‐Darstellung")

st.markdown("""
Willkommen zu dieser **App**, die finanzmathematische Grundlagen
(z. B. Barwert, Endwert, Rente, NPV) berechnet **und** den Rechenweg in
anschaulichen **Sympy/LaTeX‐Formeln** darstellt.
Außerdem werden alle Variablen kurz erklärt und die konkreten Zahlen
in die Formel eingesetzt, **noch vor** dem endgültigen Ergebnis.
""")

calc_choice = st.sidebar.radio(
    "Bitte wähle eine Berechnungsart:",
    (
        "Einmalige Zahlung (Barwert / Endwert)",
        "Renten-Berechnungen (Bar- und Endwert)",
        "Rentenhöhe aus Barwert / Endwert",
        "Zahlungsreihe (NPV / FV)",
        "Amortisationsdauer (statische Methode)"
    )
)

###############################################################################
# 1) Einmalige Zahlung (Barwert / Endwert)
###############################################################################
if calc_choice == "Einmalige Zahlung (Barwert / Endwert)":
    st.subheader("Einmalige Zahlung: Barwert und Endwert")
    tab = st.radio("Berechnung auswählen", ("Endwert", "Barwert"), horizontal=True)

    # Sympy-Symbole
    BW0Sym, iSym, nSym, EWnSym = sympy.symbols("BW_0 i n EW_n", positive=True)

    if tab == "Endwert":
        # Symbolische Formel: EW_n = BW_0*(1+i)^n
        endwert_expr = BW0Sym * (1 + iSym)**nSym

        pv_val = st.number_input("Aktueller Betrag (BW₀)", value=1000.0, step=100.0)
        i_val = st.number_input("Zinssatz i (z. B. 0.06 für 6%)", value=0.06)
        n_val = st.number_input("Anzahl Perioden (n)", value=8, step=1)

        if st.button("Berechne Endwert"):
            result = future_value(pv_val, i_val, n_val)
            show_sympy_formula(
                expr=endwert_expr,
                symbols_map={BW0Sym: pv_val, iSym: i_val, nSym: n_val},
                result_label="EW_n",
                numeric=result
            )
            st.success(f"Endwert (EWₙ) = {result:,.2f} €")

            df_chart = visualize_single_payment_growth(pv_val, i_val, n_val)
            st.line_chart(df_chart)

    else:
        # Symbolische Formel: BW_0 = EW_n / (1+i)^n
        barwert_expr = EWnSym / ((1 + iSym)**nSym)

        fv_val = st.number_input("Zukünftiger Betrag (EWₙ)", value=1000.0, step=100.0)
        i_val = st.number_input("Zinssatz i (z. B. 0.06 für 6%)", value=0.06)
        n_val = st.number_input("Anzahl Perioden (n)", value=8, step=1)

        if st.button("Berechne Barwert"):
            result = present_value(fv_val, i_val, n_val)
            show_sympy_formula(
                expr=barwert_expr,
                symbols_map={EWnSym: fv_val, iSym: i_val, nSym: n_val},
                result_label="BW_0",
                numeric=result
            )
            st.success(f"Barwert (BW₀) = {result:,.2f} €")


###############################################################################
# 2) Renten-Berechnungen (Bar- und Endwert)
###############################################################################
elif calc_choice == "Renten-Berechnungen (Bar- und Endwert)":
    st.subheader("Renten-Berechnungen (gleichbleibende Zahlungen b)")
    tab = st.radio("Berechnung auswählen", ("Rentenbarwert", "Rentenendwert"), horizontal=True)

    bSym, iSym, nSym = sympy.symbols("b i n", positive=True)
    BW0Sym, EWnSym = sympy.symbols("BW_0 EW_n", positive=True)

    if tab == "Rentenbarwert":
        # BW_0 = b * ((1+i)^n - 1) / [ i*(1+i)^n ]
        barwert_expr = bSym * ((1 + iSym)**nSym - 1)/(iSym*(1 + iSym)**nSym)

        b_val = st.number_input("Jährliche Rentenzahlung b", value=1000.0, step=100.0)
        i_val = st.number_input("Zinssatz i", value=0.06)
        n_val = st.number_input("Anzahl Perioden n", value=8, step=1)

        if st.button("Berechne Renten-Barwert"):
            result = annuity_present_value(b_val, i_val, n_val)
            show_sympy_formula(
                expr=barwert_expr,
                symbols_map={bSym: b_val, iSym: i_val, nSym: n_val},
                result_label="BW_0",
                numeric=result
            )
            st.success(f"Barwert der Rente = {result:,.2f} €")
            
            df_barwert = visualize_annuity_present(b_val, i_val, n_val)
            st.line_chart(df_barwert)

    else:
        # EW_n = b * [((1 + i)^n - 1)/i]
        endwert_expr = bSym * (((1 + iSym)**nSym) - 1)/iSym

        b_val = st.number_input("Jährliche Rentenzahlung b", value=1000.0, step=100.0)
        i_val = st.number_input("Zinssatz i", value=0.06)
        n_val = st.number_input("Anzahl Perioden n", value=8, step=1)

        if st.button("Berechne Renten-Endwert"):
            result = annuity_future_value(b_val, i_val, n_val)
            show_sympy_formula(
                expr=endwert_expr,
                symbols_map={bSym: b_val, iSym: i_val, nSym: n_val},
                result_label="EW_n",
                numeric=result
            )
            st.success(f"Endwert der Rente = {result:,.2f} €")

            df_endwert = visualize_annuity_future(b_val, i_val, n_val)
            st.line_chart(df_endwert)


###############################################################################
# 3) Rentenhöhe aus Barwert / Endwert
###############################################################################
elif calc_choice == "Rentenhöhe aus Barwert / Endwert":
    st.subheader("Gesuchte Rentenhöhe b aus gegebenem Barwert oder Endwert")
    tab = st.radio("Was ist vorgegeben?", ("Barwert (BW₀)", "Endwert (EWₙ)"), horizontal=True)

    bSym, BW0Sym, EWnSym, iSym, nSym = sympy.symbols("b BW_0 EW_n i n", positive=True)

    # b = BW_0 * [ i*(1+i)^n / ((1+i)^n - 1) ]
    b_from_BW_expr = BW0Sym * (iSym*(1+iSym)**nSym)/((1+iSym)**nSym - 1)

    # b = EW_n * [ i / ((1+i)^n - 1) ]
    b_from_EW_expr = EWnSym * (iSym / ((1+iSym)**nSym - 1))

    if tab == "Barwert (BW₀)":
        pv_val = st.number_input("Barwert (BW₀)", value=6209.79, step=100.0)
        i_val = st.number_input("Zinssatz i", value=0.06)
        n_val = st.number_input("Anzahl Perioden n", value=8, step=1)

        if st.button("Berechne b aus Barwert"):
            b_stern = annuity_from_present_value(pv_val, i_val, n_val)
            show_sympy_formula(
                expr=b_from_BW_expr,
                symbols_map={BW0Sym: pv_val, iSym: i_val, nSym: n_val},
                result_label="b",
                numeric=b_stern
            )
            st.success(f"Rentenhöhe b = {b_stern:,.2f} €")

    else:
        fv_val = st.number_input("Endwert (EWₙ)", value=9897.47, step=100.0)
        i_val = st.number_input("Zinssatz i", value=0.06)
        n_val = st.number_input("Anzahl Perioden n", value=8, step=1)

        if st.button("Berechne b aus Endwert"):
            b_stern = annuity_from_future_value(fv_val, i_val, n_val)
            show_sympy_formula(
                expr=b_from_EW_expr,
                symbols_map={EWnSym: fv_val, iSym: i_val, nSym: n_val},
                result_label="b",
                numeric=b_stern
            )
            st.success(f"Rentenhöhe b = {b_stern:,.2f} €")


###############################################################################
# 4) Zahlungsreihe (NPV / FV)
###############################################################################
elif calc_choice == "Zahlungsreihe (NPV / FV)":
    st.subheader("Zahlungsreihe (unregelmäßige Zahlungen) – NPV / FV")
    st.markdown("""
    Gib hier eine beliebige Reihe an jährlichen **Cashflows** ein.
    Dann kannst du den **Barwert (NPV)** oder **Endwert (FV)** berechnen.
    """)

    cf_str = st.text_input("Cashflows als Komma-getrennte Liste (z.B. '100, 200, 300'):", "100,200,300")
    i_val = st.number_input("Zinssatz i", value=0.06)
    tab = st.radio("Was möchtest du berechnen?", ("Barwert (NPV)", "Endwert (FV)"), horizontal=True)

    # Für die Summenformel haben wir symbolische Ausdrücke, 
    # aber hier reicht die statische Demonstration, da wir beliebig viele CFs haben.
    if st.button("Berechnen"):
        try:
            cashflows = [float(x.strip()) for x in cf_str.split(",")]
        except ValueError:
            st.error("Fehler beim Einlesen der Cashflows. Bitte gültige Zahlen im CSV-Format eingeben.")
            st.stop()

        if tab == "Barwert (NPV)":
            result = npv_of_cashflow_series(cashflows, i_val)

            st.markdown("**Symbolische Formel** (für n Zahlungen):")
            st.latex(
                r"\text{NPV} = \sum_{t=1}^n \frac{c_t}{(1 + i)^t}"
            )
            st.write(f"Cashflows: {cashflows}, Zinssatz i={i_val}")
            st.latex(f"\\text{{NPV}} \\approx {result:,.2f}")

            st.success(f"Barwert (NPV) = {result:,.2f} €")
            df_cf = visualize_cashflow_series(cashflows)
            st.bar_chart(df_cf)

        else:
            result = fv_of_cashflow_series(cashflows, i_val)

            st.markdown("**Symbolische Formel** (für n Zahlungen):")
            st.latex(
                r"\text{FV} = \sum_{t=1}^n c_t \,(1 + i)^{(n - t)}"
            )
            st.write(f"Cashflows: {cashflows}, Zinssatz i={i_val}")
            st.latex(f"\\text{{FV}} \\approx {result:,.2f}")

            st.success(f"Endwert (FV) = {result:,.2f} €")
            df_cf = visualize_cashflow_series(cashflows)
            st.bar_chart(df_cf)


###############################################################################
# 5) Amortisationsdauer (statische Methode)
###############################################################################
else:
    st.subheader("Statische Amortisationsdauer (Payback Period)")

    # Symbolische Formel: AD = a_0 / C
    a0Sym, CSym, ADSym = sympy.symbols("a_0 C AD", positive=True)
    ad_expr = a0Sym / CSym

    a0 = st.number_input("Anschaffungsausgabe a₀", value=10000.0, step=500.0)
    c  = st.number_input("Jährlicher Rückfluss C", value=2000.0, step=500.0)
    show_chart = st.checkbox("Grafik anzeigen?")

    if st.button("Berechnen"):
        ad = payback_period(a0, c)
        if ad == float("inf"):
            st.error("Amortisationsdauer = ∞ (keine Amortisation, da jährlicher Rückfluss = 0).")
        else:
            show_sympy_formula(
                expr=ad_expr,
                symbols_map={a0Sym: a0, CSym: c},
                result_label="AD",
                numeric=ad
            )
            st.success(f"Die statische Amortisationsdauer beträgt {ad:,.2f} Jahre.")

            if show_chart:
                chart = plot_payback(a0, c)
                st.altair_chart(chart, use_container_width=True)
                st.info(
                    "Die rote Linie markiert die Anfangsinvestition a₀. "
                    "Dort, wo die blaue Kurve sie schneidet, ist das 'Break Even'-Jahr."
                )

st.markdown("---")
st.caption("© 2025 – Ausführliche Formeln, Variablen‐Erklärungen und Visualisierungen.")