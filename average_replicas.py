#!/usr/bin/env python3
# McLics - Monte Carlo Liquid Crystal Simulator
# Copyright (C) 2026 The McLics authors (see AUTHORS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""
average_replicas.py — Combina réplicas independentes de po.dat
================================================================
Para sistemas com paisagem de energia degenerada (ex.: gota nemática com
ancoramento homeotrópico numa esfera), uma única rodada de Monte Carlo pode
"congelar" em qualquer uma de várias texturas de defeito com energia local
parecida, mas parâmetro de ordem global S bem diferente. Uma rodada isolada
não é uma boa estimativa do comportamento típico do sistema.

Este script combina N réplicas independentes (mesmo input_parameters.txt,
apenas `seed` diferente) em uma única curva com estatística correta.

Uso:
    python3 average_replicas.py run_seed1/po.dat run_seed2/po.dat ... -o po_avg.dat
    python3 average_replicas.py replicas/seed*/po.dat -o po_avg.dat --txt

Como gerar as réplicas:
    Copie input_parameters.txt para N pastas, mudando só a linha `seed`
    (1, 2, 3, ...), rode mclics em cada uma, depois combine os po.dat aqui.

Estatística:
    Para cada T, combina as réplicas usando a lei da variância total:
        S_médio      = média(S_r)                       sobre as réplicas r
        Var(S) total = média(varS_r)  +  var(S_r)
                       [variância "dentro" de cada réplica] + [variância "entre" réplicas]
    O segundo termo é o que se perde ao olhar só para uma rodada — é
    exatamente a variabilidade de "qual textura de defeito se formou".
"""

import sys
import argparse
import numpy as np


def read_po_dat(path):
    header = None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not line[0].replace('.', '').replace('-', '').isdigit():
                header = line
                continue
            parts = line.split()
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue
    if not rows:
        print(f"  [ERRO] Nenhum dado válido em {path}")
        sys.exit(1)
    return header, rows


def combine(paths):
    headers = []
    per_file = []
    for p in paths:
        header, rows = read_po_dat(p)
        headers.append(header)
        per_file.append(rows)

    ncols = len(per_file[0][0])
    if not all(len(r) == ncols for rows in per_file for r in rows):
        print("  [ERRO] Réplicas com número de colunas diferente (ex.: uma usa potencial 'pear' e outra não).")
        sys.exit(1)

    lengths = [len(rows) for rows in per_file]
    if len(set(lengths)) != 1:
        print(f"  [AVISO] Réplicas com número de pontos T diferente: {lengths}")
        print("  [AVISO] Usando apenas os primeiros min(N) pontos de cada réplica.")
    n_points = min(lengths)

    combined = []
    for i in range(n_points):
        T_vals = [rows[i][0] for rows in per_file]
        if max(T_vals) - min(T_vals) > 1e-3:
            print(f"  [AVISO] Ponto {i}: temperaturas não batem entre réplicas ({T_vals}) — pulando.")
            continue
        T = np.mean(T_vals)

        S_r = np.array([rows[i][1] for rows in per_file])
        varS_r = np.array([rows[i][2] for rows in per_file])
        E_r = np.array([rows[i][3] for rows in per_file])
        varE_r = np.array([rows[i][4] for rows in per_file])

        S_mean = S_r.mean()
        # Lei da variância total: Var[S] = E[Var(S|réplica)] + Var[E(S|réplica)]
        S_var_total = varS_r.mean() + S_r.var(ddof=0)

        E_mean = E_r.mean()
        E_var_total = varE_r.mean() + E_r.var(ddof=0)

        row = [T, S_mean, S_var_total, E_mean, E_var_total]

        # Extra columns (P, dtheta when potential = pear): plain mean over replicas.
        for c in range(5, ncols):
            row.append(np.mean([rows[i][c] for rows in per_file]))

        combined.append(row)

    return headers[0] or "T S varS E varE", combined


def main():
    parser = argparse.ArgumentParser(
        description="Combina réplicas independentes de po.dat (mesmos parâmetros, seeds diferentes).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("inputs", nargs="+", help="Arquivos po.dat de cada réplica")
    parser.add_argument("-o", "--output", default="po_avg.dat", help="Arquivo de saída (padrão: po_avg.dat)")
    parser.add_argument("--txt", action="store_true", help="Imprime a tabela combinada no terminal")
    args = parser.parse_args()

    if len(args.inputs) < 2:
        print("  [AVISO] Só uma réplica foi passada — sem réplicas independentes não há como estimar")
        print("  a variabilidade entre texturas. O resultado será idêntico ao po.dat de entrada.")

    header, combined = combine(args.inputs)

    with open(args.output, "w") as f:
        f.write(header + "\n")
        for row in combined:
            f.write(" ".join(f"{v:g}" for v in row) + "\n")

    print(f"\n  {len(args.inputs)} réplicas combinadas -> {args.output}")
    print(f"  {len(combined)} pontos de T\n")

    if args.txt:
        print(f"  {'T':>8} {'S':>10} {'varS':>12} {'E':>10} {'varE':>12}")
        for row in combined:
            print(f"  {row[0]:>8.4f} {row[1]:>10.4f} {row[2]:>12.4g} {row[3]:>10.4f} {row[4]:>12.4g}")
        print()


if __name__ == "__main__":
    main()
