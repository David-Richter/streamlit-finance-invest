import streamlit as st
import pandas as pd
import altair as alt
import math
import sympy
import numpy as np
import matplotlib.pyplot as plt


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

def capital_value(a0: float, future_flows: list[float], i: float) -> float:
    """
    Berechnet den Kapitalwert einer Investition gemäß:
    
        C0 = -a0 + sum( c_t / (1 + i)^t ),  für t = 1..n
    
    Hier:
      a0           Anfangsauszahlung (positiver Wert)
      future_flows Liste der erwarteten Einzahlungen c_t für t=1..n (oder Ein-/Auszahlungen)
      i            Kalkulationszinssatz (Dezimal, z.B. 0.06)
    
    Rückgabewert: Kapitalwert C0
    """
    npv = -a0  # anfängliche Ausgabe
    for t, cf in enumerate(future_flows, start=1):
        npv += cf / ((1 + i)**t)
    return npv

def plot_investment_flows(a0: float, c_flows: list[float]):
    """
    Balkendiagramm der gesamten Zahlungsreihe:
      t=0 => -a0,
      t=1..n => c_flows
    """
    data = [{"t": 0, "Zahlung": -a0}]
    for t, cf in enumerate(c_flows, start=1):
        data.append({"t": t, "Zahlung": cf})
    df = pd.DataFrame(data)
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("t:O", title="Periode (t)"),
        y=alt.Y("Zahlung:Q", title="Zahlung"),
        tooltip=["t", "Zahlung"]
    ).properties(title="Zahlungsreihe (Investition)")
    return chart

def solve_irr(a0: float, cashflows: list[float]) -> float:
    """
    Sucht den internen Zinsfuß r numerisch aus:
        0 = -a0 + sum( c_t / (1 + r)^t )
    Nutzt sympy nsolve. Gibt NaN zurück, wenn keine Lösung gefunden wird.
    """
    r = sympy.Symbol("r", real=True)
    eq_expr = -a0
    for t, cf in enumerate(cashflows, start=1):
        eq_expr += cf / ((1 + r)**t)

    guess = 0.1  # Startwert für nsolve
    try:
        sol = sympy.nsolve(eq_expr, r, guess)
        return float(sol)
    except:
        return float("nan")

def dynamic_amortization_time(a0, c, i):
    """
    Bestimmt die exakte dynamische Amortisationsdauer t' mit der Formel:
       a0 = c * [((1 + i)^(t') - 1) / (i * (1 + i)^(t'))]
    Aufgelöst nach t':
       t' = ln( 1 / [1 - (a0 * i / c)] ) / ln(1 + i)

    Parameter:
    -----------
    a0 : float
        Anfangsauszahlung (positiver Betrag, z.B. 60 für -60 in t0).
    c  : float
        Konstante Einzahlungsüberschüsse (pro Periode).
    i  : float
        Kalkulationszinssatz (z.B. 0.06 für 6%).

    Rückgabe:
    ---------
    float
        Die exakte (ggf. nicht ganzzahlige) Amortisationsdauer t'.
        Falls keine sinnvolle Lösung existiert, wird 'math.nan' zurückgegeben.
    """
    # Ausdruck umformen: a0 * i / c = 1 - (1 + i)^(-t')
    # => (1 + i)^(-t') = 1 - (a0 * i / c)
    # => Falls dieses 'alpha' <= 0 oder >= 1, gibt es keine reelle Lösung
    alpha = 1 - (a0 * i / c)
    if alpha <= 0 or alpha >= 1:
        # Keine (positive) reelle Lösung für t'
        return float('nan')

    t_prime = math.log(1.0 / alpha) / math.log(1.0 + i)
    return t_prime

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

def visualize_growth_single_payment(pv: float, i: float, n: int):
    """
    Kleines Beispiel: Zeigt die (aufgezins­ten) Werte einer einmaligen Zahlung PV
    über n Perioden bei Zinssatz i.
    """
    data = []
    for t in range(n+1):
        value = pv * (1 + i)**t
        data.append({"Jahr": t, "Wert": value})
    df = pd.DataFrame(data)
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x="Jahr:Q",
            y="Wert:Q",
            tooltip=["Jahr", "Wert"]
        )
        .properties(title="Zeitlicher Verlauf einer Einmalzahlung (aufgezinst)")
        .interactive()
    )
    return chart

def visualize_cashflow_series_example():
    """
    Einfaches Balkendiagramm für eine (fiktive) Zahlungsreihe c1..c5.
    """
    # Beispielwerte
    cashflows = [-1000, 200, 300, 400, 600]  # 5 Perioden
    data = []
    for t, cf in enumerate(cashflows, start=0):
        data.append({"Periode": t, "Zahlung c_t": cf})
    df = pd.DataFrame(data)
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Periode:O", title="Periode (t)"),
            y=alt.Y("Zahlung c_t:Q", title="Höhe der Zahlung"),
            tooltip=["Periode", "Zahlung c_t"]
        )
        .properties(title="Beispiel-Zahlungsreihe mit positiven/negativen Zahlungen")
    )
    return chart

def visualize_rente_example(b: float, i: float, n: int):
    """
    Zeigt, wie sich eine Rente b in jedem Jahr zum Endwert aufaddiert:
    Jedes Jahr Rente + Aufzinsung.
    """
    data = []
    balance = 0.0
    q = 1 + i
    for year in range(1, n+1):
        balance = balance * q + b
        data.append({"Jahr": year, "Kumulierte Summe": balance})
    df = pd.DataFrame(data)
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x="Jahr:Q",
            y="Kumulierte Summe:Q",
            tooltip=["Jahr", "Kumulierte Summe"]
        )
        .properties(title="Beispiel: Rente b, jährlich aufgezinst bis Endwert")
        .interactive()
    )
    return chart

def visualize_cashflow_series(cashflows: list[float]):
    df = pd.DataFrame({
        "Jahr": list(range(1, len(cashflows)+1)),
        "Cashflow": cashflows
    }).set_index("Jahr")
    return df

def npv_for_rate(a0: float, cflows: list[float], rate: float) -> float:
    """
    Hilfsfunktion: Kapitalwert bei Zinssatz = rate
    """
    npv_val = -a0
    for t, cf in enumerate(cflows, start=1):
        npv_val += cf / ((1 + rate)**t)
    return npv_val

def plot_capital_value_curve(a0: float, cflows: list[float], max_rate: float = 0.5, steps=100):
    """
    Erzeugt ein Liniendiagramm, das C0(i) für i in [0..max_rate] darstellt.
    steps = Anzahl diskreter i-Werte.
    
    Der Schnittpunkt mit 0 (falls vorhanden) entspricht dem IRR.
    """
    data = []
    for i in [x*(max_rate/steps) for x in range(steps+1)]:
        c0_i = npv_for_rate(a0, cflows, i)
        data.append({"Zinssatz i": i, "Kapitalwert": c0_i})

    df = pd.DataFrame(data)
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X("Zinssatz i:Q", title="Diskontierungszinssatz i"),
        y=alt.Y("Kapitalwert:Q", title="Kapitalwert C₀(i)"),
        tooltip=["Zinssatz i", "Kapitalwert"]
    ).properties(title="Kapitalwertkurve C₀(i)")

    # Optional: horizontale Linie bei Kapitalwert=0
    rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red").encode(y="y")

    return (chart + rule).interactive()

def plot_cumulative_discounted_flows(a0, c, i, max_periods=10):
    """
    Plottet den kumulierten Barwert der Rückflüsse bis zu `max_periods`.
    So sieht man in welcher Periode der anfängliche Kapitaleinsatz 'a0' gedeckt ist.

    Parameter:
    ----------
    a0 : float
        Anfangsauszahlung (positiver Betrag)
    c  : float
        Konstante Rückflüsse pro Periode
    i  : float
        Zinssatz
    max_periods : int
        Anzahl Perioden, die in der Grafik betrachtet werden sollen
    """
    # Diskontierungsfaktor (1 + i)^(-t)
    discounted_sum = 0.0
    cumulated_values = []

    # t läuft von 1 bis max_periods
    for t in range(1, max_periods + 1):
        # Barwert des Cashflows in Periode t
        pv = c / ((1 + i) ** t)
        discounted_sum += pv
        cumulated_values.append(discounted_sum)

    # Plot
    plt.figure(figsize=(7,4), dpi=100)
    periods = np.arange(1, max_periods + 1)

    # Kumulierte diskontierte Rückflüsse als Kurve
    plt.plot(periods, cumulated_values, marker='o', label='Kumulierte Barwerte')

    # Horizontale Linie bei 'a0' => zeigt, wann Amortisation erreicht
    plt.axhline(y=a0, color='r', linestyle='--', label='Anfangsauszahlung a0')

    plt.title('Kumulierte diskontierte Rückflüsse vs. Anfangsauszahlung')
    plt.xlabel('Periode t')
    plt.ylabel('Kumulierte Barwerte')
    plt.legend()
    plt.grid(True)
    plt.show()

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
    "AD":   "Amortisationsdauer (Jahre)",
    "r":    "Interner Zinsfuß (gesuchte Rendite)"
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
        "Amortisationsdauer (statische Methode)",
        "Kapitalwert",
        "Interner Zinsfuß",
        "Dynamische Amortisationsdauer",
        "Theoretisches Wissen"
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
elif calc_choice == "Amortisationsdauer (statische Methode)":
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


###############################################################################
# 6) Kapitalwert (Nettobarwert)
###############################################################################
elif calc_choice == "Kapitalwert":
    def show_sympy_formula(expr, symbols_map: dict, result_label="Ergebnis", numeric=None):
        """
        Zeigt die allgemeine Formel, die Substitution und das numerische Ergebnis
        in schöner LaTeX-Form an.
        """
        from sympy import latex

        # 1) Symbolische Formel
        st.markdown("**Allgemeine Formel**:")
        st.latex(f"{result_label} = {latex(expr)}")
        
        # 2) Eingesetzte Werte
        substituted_expr = expr.subs(symbols_map)
        st.markdown("**Eingesetzte Werte in der Formel**:")
        st.latex(f"{result_label} = {latex(substituted_expr)}")

        # 3) Numerisches Ergebnis
        if numeric is not None:
            st.markdown("**Numerisches Ergebnis**:")
            st.latex(f"{result_label} \\approx {numeric:,.2f}")


    st.subheader("Kapitalwert (C₀) mit flexibler Zahlungsreihe")

    st.markdown(r"""
    **Kurze Erklärung**  
    Der Kapitalwert \(C_0\) (auch Nettobarwert) bezeichnet den
    Barwert sämtlicher Ein- und Auszahlungen einer Investition zum Zeitpunkt \(t = 0\).

    Formel (flexible Anzahl an Zahlungen):
    \[
        C_0 = -a_0 \;+\; \sum_{t=1}^n \frac{c_t}{(1 + i)^t}.
    \]

    - \(a_0\): Anfangsauszahlung (positiver Wert eingegeben, wird als Abfluss verbucht)
    - \(c_t\): erwarteter Cashflow in Periode \(t\)
    - \(i\): Kalkulationszinssatz (dezimal, z.B. 0.06)
    - \(n\): Anzahl zukünftiger Perioden (Länge der Liste)

    **Akzeptanzkriterium**:  
    - \(C_0 \ge 0\) ⇒ Investition ist vorteilhaft (oder mindestens genauso gut wie Alternativen).
    """)

    # Eingaben
    a0_val = st.number_input("Anfangsauszahlung a₀ (positiv)", value=1000.0, step=100.0)
    i_val  = st.number_input("Kalkulationszinssatz i (z.B. 0.06)", value=0.06)
    
    cflows_str = st.text_input(
        "Zukünftige Zahlungen c₁, c₂, ... (Komma-getrennt)",
        value="400,600,800"
    )

    if st.button("Kapitalwert berechnen"):
        # 1) Liste der Cashflows parsen

        future_flows = [float(x.strip()) for x in cflows_str.split(",")]

        n = len(future_flows)
        # 2) Symbolische Variablen definieren
        #    a_0 und i definieren wir einzeln,
        #    c_1..c_n werden dynamisch generiert:
        a0Sym, iSym = sympy.symbols("a_0 i", positive=True)
        # Erzeuge c1, c2, ..., cN
        cSymbols = sympy.symbols(" ".join(f"c_{t}" for t in range(1, n+1)), real=True)

        # 3) Kapitalwert-Formel dynamisch aufbauen
        # Start mit -a_0
        capital_value_expr = -a0Sym
        # + sum( c_t / (1+i)^t ) 
        for t in range(n):
            capital_value_expr += cSymbols[t] / ((1 + iSym)**(t+1))

        # 4) Dictionary für subs()
        subs_map = {
            a0Sym: a0_val,
            iSym: i_val,
        }
        for t, cf in enumerate(future_flows):
            subs_map[cSymbols[t]] = cf

        # 5) Symbolisch auswerten
        c0_sympy = capital_value_expr.subs(subs_map)
        c0_value = float(c0_sympy.evalf())

        # 6) Schöne Ausgabe
        show_sympy_formula(
            expr=capital_value_expr,
            symbols_map=subs_map,
            result_label="C_0",
            numeric=c0_value
        )

        st.success(f"Kapitalwert C₀ = {c0_value:,.2f} €")

        # 7) Visualisierung der gesamten Zahlungsreihe (inkl. -a0 in Periode 0)
        data = []
        data.append({"Periode": 0, "Zahlung": -a0_val})
        for idx, cf in enumerate(future_flows, start=1):
            data.append({"Periode": idx, "Zahlung": cf})

        df = pd.DataFrame(data)
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("Periode:O"),
                y=alt.Y("Zahlung:Q"),
                tooltip=["Periode", "Zahlung"]
            )
            .properties(title="Zahlungsreihe")
        )
        st.altair_chart(chart, use_container_width=True)

        st.info("Periode 0: Anschaffung (negativ), Periode 1..n: Rückflüsse.")


###############################################################################
# 6) Interner Zinsfuß
###############################################################################

# elif calc_choice == "Interner Zinsfuß":
#         st.subheader("Interner Zinsfuß (IRR) – Vergleich mit Kapitalmarktzins")

#         st.markdown(r"""
#         Der \textbf{interne Zinsfuß} (engl.\ IRR) ist der Zinssatz \(r\), bei dem der 
#         Kapitalwert einer Investition gleich Null wird.  
#         \[
#         0 = -a_0 + \sum_{t=1}^n \frac{c_t}{(1 + r)^t}.
#         \]
#         **Vergleich mit Kapitalmarktzins** \(i_{\mathrm{ref}}\):  
#         - Wenn \(r > i_{\mathrm{ref}}\), ist die Investition \textit{vorteilhaft}.  
#         - Wenn \(r = i_{\mathrm{ref}}\), ist man \textit{indifferent}.  
#         - Wenn \(r < i_{\mathrm{ref}}\), ist die Investition \textit{nicht vorteilhaft}.  
#         """)

#         # Eingaben:
#         a0 = st.number_input("Anfangsauszahlung a₀", value=1000.0)
#         cflows_str = st.text_input("Zukünftige Cashflows, Komma-getrennt", "400,600,800")
#         # max_rate = st.number_input("Obergrenze für i in der Grafik (z.B. 0.50 = 50%)", value=0.50)
#         max_rate = st.number_input("Kapitalzins", value=0.50)
#         steps = st.slider("Anzahl Schritte in der Kurve", 10, 200, 100)

#         if st.button("IRR berechnen"):
#             #flows = [float(x.strip()) for x in cf_str.split(",")]
#             flows = [float(x.strip()) for x in cflows_str.split(",")]
#             cflows = [float(x.strip()) for x in cflows_str.split(",")]

#             # 1) IRR numerisch lösen
#             irr = solve_irr(a0, flows)

#             # 2) Symbolische Darstellung
#             rSym = sympy.Symbol("r", real=True)
#             eq_expr = -sympy.Symbol("a_0")
#             for t, cf in enumerate(flows, start=1):
#                 eq_expr += sympy.Symbol(f"c_{t}", real=True) / (1 + rSym)**t

#             subs_map = {sympy.Symbol("a_0"): a0}
#             for t, cf in enumerate(flows, start=1):
#                 subs_map[sympy.Symbol(f"c_{t}", real=True)] = cf

#             show_sympy_formula(eq_expr, subs_map, result_label="NPV")
#             st.write("Hier suchen wir den Zinssatz r, sodass NPV=0.")

#             if math.isnan(irr):
#                 st.error("Keine eindeutige Lösung gefunden (evtl. mehrere oder keine Nullstellen).")
#             else:
#                 st.success(f"Interner Zinsfuß IRR = {irr*100:,.2f} % p.a.")
                
#             # 3) Vergleich IRR vs i_ref
#             if abs(irr - max_rate) < 1e-7:
#                 st.info("IRR ≈ Vergleichszins → Indifferenz.")
#             elif irr > max_rate:
#                 st.success("IRR > Vergleichszins → **Investition ist vorteilhaft**.")
#             else:
#                 st.warning("IRR < Vergleichszins → **Investition nicht vorteilhaft**.")

#             # 4) Visualisierung
#             chart = plot_investment_flows(a0, flows)
#             st.altair_chart(chart, use_container_width=True)
#             st.info("Darstellung der gesamten Zahlungsreihe: Periode 0 => -a₀, Periode 1..n => cₜ.")

#             # Plot
#             chart = plot_capital_value_curve(a0, cflows, max_rate=max_rate, steps=steps)
#             st.altair_chart(chart, use_container_width=True)

#             st.info("""
#             Die rote Linie markiert den Kapitalwert = 0.
#             Dort, wo die Kurve sie schneidet (C₀(i)=0), liegt der interne Zinsfuß r.
#             """)

elif calc_choice == "Interner Zinsfuß":
    st.subheader("Interner Zinsfuß (IRR) – Vergleich mit Kapitalmarktzins")

    st.markdown(r"""
    Der \textbf{interne Zinsfuß} (engl.\ IRR) ist der Zinssatz \(r\), bei dem der 
    Kapitalwert einer Investition gleich Null wird.  
    \[
    0 = -a_0 + \sum_{t=1}^n \frac{c_t}{(1 + r)^t}.
    \]
    **Vergleich mit Kapitalmarktzins** \(i_{\mathrm{ref}}\):  
    - Wenn \(r > i_{\mathrm{ref}}\), ist die Investition \textit{vorteilhaft}.  
    - Wenn \(r = i_{\mathrm{ref}}\), ist man \textit{indifferent}.  
    - Wenn \(r < i_{\mathrm{ref}}\), ist die Investition \textit{nicht vorteilhaft}.  
    """)

    # Eingaben:
    a0 = st.number_input("Anfangsauszahlung a₀", value=1000.0)
    cflows_str = st.text_input("Zukünftige Cashflows, Komma-getrennt", "400,600,800")

    # Kapitalmarktzins (Referenzzinssatz) als eigener Input:
    i_ref = st.number_input(
        "Kapitalmarktzins i_ref (z.B. 0.06 = 6%)",
        value=0.06,
        min_value=0.00,
        max_value=1.00,
        step=0.01,
        format="%.4f"
    )

    # Obergrenze für die Grafik separat:
    # max_plot_rate = st.number_input(
    #     "Obergrenze für i in der Grafik (z.B. 0.50 = 50%)",
    #     value=0.50,
    #     min_value=0.00,
    #     max_value=1.00,
    #     step=0.05,
    #     format="%.2f"
    # )

    # steps = st.slider("Anzahl Schritte in der Kurve", 10, 200, 100)

    if st.button("IRR berechnen"):

        # Zahlungsreihe (nur einmal nötig):
        flows = [float(x.strip()) for x in cflows_str.split(",")]

        # 1) IRR numerisch lösen
        irr = solve_irr(a0, flows)  # <-- Annahme: solve_irr ist korrekt implementiert

        # 2) Symbolische Darstellung:
        rSym = sympy.Symbol("r", real=True)
        eq_expr = -sympy.Symbol("a_0")
        for t, cf in enumerate(flows, start=1):
            eq_expr += sympy.Symbol(f"c_{t}", real=True) / (1 + rSym)**t

        subs_map = {sympy.Symbol("a_0"): a0}
        for t, cf in enumerate(flows, start=1):
            subs_map[sympy.Symbol(f"c_{t}", real=True)] = cf

        show_sympy_formula(eq_expr, subs_map, result_label="NPV")
        st.write("Hier suchen wir den Zinssatz r, sodass NPV=0.")

        if math.isnan(irr):
            st.error("Keine eindeutige Lösung gefunden (evtl. mehrere oder keine Nullstellen).")
        else:
            st.success(f"Interner Zinsfuß IRR = {irr*100:,.2f} % p.a.")

        # 3) Vergleich IRR vs i_ref
        if math.isnan(irr):
            pass  # Kein Vergleich möglich
        elif abs(irr - i_ref) < 1e-7:
            st.info("IRR ≈ Vergleichszins → Indifferenz.")
        elif irr > i_ref:
            st.success("IRR > Vergleichszins → **Investition ist vorteilhaft**.")
        else:
            st.warning("IRR < Vergleichszins → **Investition nicht vorteilhaft**.")

        # 4) Visualisierung der Zahlungsreihe
        chart = plot_investment_flows(a0, flows)
        st.altair_chart(chart, use_container_width=True)
        st.info("Zahlungsreihe: Periode 0 => -a₀, Periode 1..n => cₜ.")

        # Kapitalwertkurve mit eigener Obergrenze
        chart = plot_capital_value_curve(a0, flows, max_rate=1, steps=100)
        st.altair_chart(chart, use_container_width=True)
        st.info("""
        Die rote Linie markiert den Kapitalwert = 0.
        Dort, wo die Kurve sie schneidet (C₀(i)=0), liegt der interne Zinsfuß r.
        """)


###############################################################################
# 7) Dynamische Amortisationsdauer
###############################################################################

elif calc_choice == "Dynamische Amortisationsdauer":
    
    st.subheader("Dynamische Amortisationsdauer")
    # -------------------------
    # 1) Variablen anpassen
    # -------------------------
    # a0 = 60.0  # Anfangsauszahlung (Beispiel: -60 in t0)
    # c  = 14.0  # Konstante Einzahlungsüberschüsse
    # i  = 0.06  # Kalkulationszinssatz (z.B. 6%)
    # max_periods = 10  # Für die Visualisierung

    # Eingaben:
    a0 = st.number_input("Anfangsauszahlung a₀", value=60.0)
    c  = st.number_input("Konstante Einzahlungsüberschüsse", value=14.0)  # Konstante Einzahlungsüberschüsse
    i  = st.number_input("Kalkulationszinssatz (z.B. 6%)", value=0.06)  # Kalkulationszinssatz (z.B. 6%)
    max_periods = st.number_input("Max. Perioden ", value=10)  # Für die Visualisierung


    if st.button("IRR berechnen"):
        # -------------------------
        # 2) Dynamische Amortisationsdauer berechnen
        # -------------------------
        t_star = dynamic_amortization_time(a0, c, i)
        if math.isnan(t_star):
            st.warning("Keine reelle Lösung für die dynamische Amortisationsdauer (NaN).")
        else:
            st.success(f"Exakte dynamische Amortisationsdauer t' = {t_star:.3f} Perioden.")

        # -------------------------
        # 3) Visualisierung
        # -------------------------
        # chart = plot_cumulative_discounted_flows(a0, c, i, max_periods)
        # st.altair_chart(chart, use_container_width=True)

###############################################################################
# Theoretisches Wissen
###############################################################################

elif calc_choice == "Theoretisches Wissen":
    
    st.header("Theoretisches Wissen")
    topic = st.selectbox(
        "Wähle ein Themengebiet:",
        [
            "Zahlung, Zahlungsreihe, Rente",
            "Symbolerläuterungen",
            "Einmalige Zahlung",
            "Zahlungsreihe",
            "Rente",
            "Auf- und Abzinsung (Faktoren)"
        ]
    )

    if topic == "Zahlung, Zahlungsreihe, Rente":
        
        st.subheader("Was unterscheidet eine Zahlung, eine Zahlungsreihe und eine Rente?")

        st.markdown("""
        **1. Zahlung**  
        Eine einzelne Transaktion (Cashflow) zu einem bestimmten Zeitpunkt.  
        Beispiel: Eine Anschaffungszahlung von -500 € heute (t=0).

        **2. Zahlungsreihe**  
        Eine Folge mehrerer Zahlungen über die Zeit. Die einzelnen Beträge können
        sich in Höhe und Vorzeichen unterscheiden.  
        Beispiel:  
        - t=0: -500 € (Investition)  
        - t=1: +200 € (Rückfluss)  
        - t=2: +300 €

        **3. Rente**  
        Eine _gleichförmige_ Zahlungsreihe, d. h. ein fester Betrag, der regelmäßig
        (z. B. jährlich) gezahlt oder empfangen wird, meistens am Periodenende.
        Beispiel: Jedes Jahr 1.000 € über n Jahre.  
        """)

        st.info("""
        Zusammengefasst:  
        - Eine *Zahlung* bezeichnet genau **eine** Transaktion (einmalig).  
        - Eine *Zahlungsreihe* ist jede beliebige **Abfolge** mehrerer Zahlungen.  
        - Eine *Rente* ist eine **gleichmäßige** Zahlungsreihe (feste Rate, konstantes Intervall).
        """)
    
###############################################################################
# Erweiterter Theoriebereich: Finanzmathematische Grundlagen
###############################################################################
# theory_choice2 = st.sidebar.radio(
#     "Finanzmathematische Grundlagen anzeigen?",
#     ("Keine Anzeige", "Symbolerläuterung & Formeln")
# )

    if topic == "Symbolerläuterungen":
        st.header("Finanzmathematische Grundlagen")

        # 1) Symbolerläuterung
        st.subheader("Symbolerläuterung")
        st.markdown("""
        - **BW₀**: Barwert einer Zahlung oder Zahlungsreihe (heutiger Gegenwert)
        - **EWₙ**: Endwert einer Zahlung oder Zahlungsreihe im Zeitpunkt t = n
        - **b**: Rentenzahlung (jährlich gleichbleibende Zahlung)
        - **cᵗ**: Einzelne Cashflows (Zahlungen) im Zeitverlauf t = 1, 2, ..., n
        - **i**: Kalkulationszinssatz (pro Periode, als Dezimalzahl, z. B. 0.06 für 6 %)
        - **q** = (1 + i)
        - **qⁿ**: Aufzinsungsfaktor
        - **q⁻ⁿ**: Diskontierungsfaktor (= 1 / qⁿ)
        """)

        # 2) Formeln
        st.subheader("Grundlegende Formeln")

        st.markdown("**(a) Endwert einer (einmaligen) Zahlung**")
        st.latex(r"""
        EW_n = BW_0 \cdot (1 + i)^n
        """)

        st.markdown("**(b) Barwert einer (einmaligen) Zahlung**")
        st.latex(r"""
        BW_0 = \frac{EW_n}{(1 + i)^n}
        """)

        st.markdown("**(c) Endwert einer Rente (Rentenendwert)**")
        st.markdown("Diese Formel nennt man oft den **Rentenendwertfaktor**:")
        st.latex(r"""
        EW_n = b \cdot \frac{(1+i)^n - 1}{i}
        """)

        st.markdown("**(d) Barwert einer Rente (Rentenbarwert)**")
        st.markdown("Die Formel entspricht dem sog. **Rentenbarwertfaktor**:")
        st.latex(r"""
        BW_0 = b \cdot \frac{(1+i)^n - 1}{i \cdot (1+i)^n}
        """)

        st.info("""
            *Hinweis:*  
            - Eine "Rente" ist eine Zahlungsreihe mit konstanten periodischen Zahlungen b.
            - \(i\) ist der Kalkulationszinssatz pro Periode, \(n\) die Anzahl Perioden.
            - \((1 + i)^n\) nennt man Aufzinsungsfaktor, \((1 + i)^{-n}\) den Diskontierungsfaktor.
        """)
    

    if topic == "Einmalige Zahlung":
        st.subheader("Einmalige Zahlung: Barwert & Endwert")
        st.markdown("""
        Wenn du heute einen Betrag \\( BW_0 \\) anlegst, wächst er bis in n Perioden
        auf den **Endwert** \\( EW_n \\) gemäß:

        \\[
            EW_n = BW_0 \\times (1 + i)^n
        \\]

        Umgekehrt, wenn du in n Perioden \\(EW_n\\) erhältst, kannst du den heutigen
        **Barwert** \\( BW_0 \\) bestimmen:

        \\[
            BW_0 = \\frac{EW_n}{(1 + i)^n}
        \\]

        **Interpretation**:
        - \\( i \\) ist der Zinssatz p.a. (oder p.Period)
        - \\( n \\) ist die Anzahl Perioden (z. B. Jahre)
        - \\( (1 + i)^n \\) heißt **Aufzinsungsfaktor**,
          \\( (1 + i)^{-n} \\) (bzw. \\( 1/(1 + i)^n \\)) ist der **Abzinsungsfaktor**.
        """)

        st.info("""
        **Beispielvisualisierung**: 
        Unten ein kleines Diagramm, das zeigt, wie eine einmalige Zahlung
        über die Jahre aufgezinst wird.
        (Du kannst die Werte anpassen.)
        """)

        # Interaktive Felder für Demo
        pv = st.number_input("Beispiel: Anfangsbetrag (BW_0)", value=1000.0)
        i_demo = st.number_input("Zinssatz i (dezimal)", value=0.05)
        n_demo = st.slider("Anzahl Perioden n", 1, 15, 5)

        chart = visualize_growth_single_payment(pv, i_demo, n_demo)
        st.altair_chart(chart, use_container_width=True)

    elif topic == "Zahlungsreihe":
        st.subheader("Zahlungsreihe: Mehrere, ggf. unterschiedliche Zahlungen")

        st.markdown(r"""
        Wenn man nicht nur **eine** Zahlung, sondern mehrere (positive wie negative)
        über die Zeit hat, spricht man von einer **Zahlungsreihe** \(\{c_0, c_1, \dots, c_n\}\).

        - **Barwert** aller Zahlungen ist die Summe der **einzelnen Barwerte**:

          \[
            BW_0 = \sum_{t=0}^{n} \frac{c_t}{(1 + i)^t}
          \]

        - **Endwert** dieser Zahlungsreihe (Zeitpunkt n) erhält man durch
          **Aufzinsung** jeder Zahlung vom Zeitpunkt t bis n:

          \[
            EW_n = \sum_{t=0}^{n} c_t \times (1 + i)^{(n - t)}
          \]

        Damit kann man viele beliebige Cashflows (Investitions‐ und Rückflüsse)
        bewerten und vergleichen.
        """)

        st.info("""
        **Beispielvisualisierung**:
        Unten eine einfache Balkengrafik, die eine Beispielreihe
        mit einigen negativen (Auszahlungen) und positiven (Einzahlungen)
        Cashflows zeigt.
        """)

        chart = visualize_cashflow_series_example()
        st.altair_chart(chart, use_container_width=True)

    elif topic == "Rente":
        st.subheader("Rente: Gleichbleibende (jährliche) Zahlung b")

        st.markdown(r"""
        Eine **Rente** liegt vor, wenn es jedes Jahr (oder jede Periode)
        **denselben** Betrag \(b\) gibt. Damit gelten einfache Formeln:

        - **Endwert einer Rente** (nachschüssig gezahlte b):
          \[
            EW_n = b \times \frac{(1 + i)^n - 1}{i}
          \]
          (Manchmal heißt diese Formel auch *Rentenendwertfaktor*.)

        - **Barwert einer Rente**:
          \[
            BW_0 = b \times \frac{(1 + i)^n - 1}{i \times (1 + i)^n}
          \]
          (*Rentenbarwertfaktor*.)

        **Interpretation**:
        - Du zahlst (oder erhältst) jedes Jahr b, und willst wissen,
          wieviel das am Ende der Laufzeit n insgesamt ausmacht
          (aufgezinst),
        - oder welchen Wert das heute (abgezinst) hat.
        """)

        st.info("""
        **Beispielvisualisierung**:  
        Wir simulieren hier, wie sich eine jährliche Rente b
        über n Jahre kumuliert, wenn jede Zahlung sofort weiterverzinst wird.
        """)

        b_demo = st.number_input("Jährliche Rate b", value=100.0)
        i_demo = st.number_input("Zinssatz i (z.B. 0.06)", value=0.06)
        n_demo = st.slider("Anzahl Perioden n", 1, 15, 5)
        chart = visualize_rente_example(b_demo, i_demo, n_demo)
        st.altair_chart(chart, use_container_width=True)

    else:  # "Auf- und Abzinsung (Faktoren)"
        st.subheader("Auf- und Abzinsung: Wichtige Faktoren")

        st.markdown(r"""
        **Aufzinsungsfaktor**:  
        \[
          (1 + i)^n = q^n 
        \]

        **Abzinsungsfaktor**:  
        \[
          \frac{1}{(1 + i)^n} = \frac{1}{q^n} = (1 + i)^{-n}
        \]

        Diese Faktoren brauchst du:
        - Zum **Aufzinsen** (heutige Werte -> in n Jahren),
        - Zum **Abzinsen** (Zukunftswerte -> heute).  
        """)

        st.markdown(r"""
        **Rentenendwertfaktor**:
        \[
          \frac{(1 + i)^n - 1}{i}
        \]
        Damit berechnest du den **Endwert** einer nachschüssigen
        Rente (Rate b).  

        **Rentenbarwertfaktor**:
        \[
          \frac{(1 + i)^n - 1}{i \times (1 + i)^n}
        \]
        Damit berechnest du den **Barwert** einer nachschüssigen Rente.
        """)

        st.markdown(r"""
        **Kapitalwiedergewinnungsfaktor**:  
        \[
          \frac{i \times (1 + i)^n}{(1 + i)^n - 1}
        \]
        Mit diesem Faktor findest du aus einem (Barwert) eine konstante
        Rate b, die das Kapital über n Perioden "wiedergewinnt".
        """)

        st.info("""
        Die hier genannten Faktoren sind zentrale Bausteine,
        um **Barwert** und **Endwert** einer ganzen Reihe gleichförmiger
        Zahlungen zu bestimmen.
        """)


st.markdown("---")
st.caption("© 2025 – Ausführliche Formeln, Variablen‐Erklärungen und Visualisierungen.")