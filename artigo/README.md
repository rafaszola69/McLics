# Caso teste do potencial pear para o artigo do McLics

Simulações e figuras do exemplo do potencial de Dhakal & Selinger (PRE 81, 031704, 2010) no artigo do código.
O estudo completo, com a física além do necessário para o artigo, está em `../pear/` (`RESULTADOS.md`, `relatorio/`).

## Rodar

```bash
cd artigo
./run.sh               # todas as etapas; ou ./run.sh selinger | phase | phasefine | droplets
python3 make_figures.py
```

`run.sh` usa `mclics` do PATH (versão atual do código). Rodadas com `DONE` são puladas, então dá para interromper
e relançar. Variáveis: `JOBS` (rodadas simultâneas), `SEEDS` (padrão `"1 2 3"`), `DROP_SIZES`, `LINEAR_N`, `B1_VALUES`.

| Etapa | Sistema | O que mostra |
|---|---|---|
| `selinger` | bulk 20³, x,y livres, z periódico; A=1.5, B1=0.09, C=0.3; T 1.6→0.1, ΔT=0.025 | validação: sequência I → N → polar (condições da Fig. 2a do paper) |
| `phase` | mesmo bulk, B1 = 0, 0.09, 0.3, 0.6, 1.0, 1.5; T 3.0→0.1 | nemático intermediário para B1 fraco, transição direta I → polar para B1 forte |
| `phasefine` | mesmo bulk na grade de B do diagrama de fases de Dhakal & Selinger (B1 = 0.02 … 0.76), T 2.0→0.04, ΔT=0.02 | comparação direta com o diagrama do paper (sobreposto no painel b) |
| `droplets` | gotas homeotrópicas (W=1.5) N=21, 31, 41 com C=+0.3, −0.3, 0; N=31 com C=0.1, 0.2; T 1.4→0.1 | polarização radial induzida pelo splay do ouriço, simetria em C, linearidade, p(r)=K/r |

Total com 3 seeds: 3 (selinger) + 18 (phase) + 48 (phasefine) + 33 (droplets) = 102 rodadas, todas de tamanho ≤ 41³.
Se `runs/phasefine` tiver rodadas concluídas, a figura usa essa etapa no lugar de `phase`.

## Saída

- `figs/fig_pear_bulk.pdf` — (a) S e P vs T; (b) temperaturas de transição vs B1.
- `figs/fig_pear_droplet.pdf` — (a,b) cortes do ouriço para C=±0.3; (c) P_r vs T; (d) P_r/C.
- `figs/fig_pear_profile.pdf` — (a) p(r) para três raios; (b) r·p(r).
- `figures.tex` — ambientes `figure*` com legendas em inglês; os números das legendas vêm dos dados.
- `figs/numbers.txt` — os números usados nas legendas, para conferência.

Opções do `make_figures.py`: `--profile-T` (0.75), `--slice-T` (0.1), `--slice-N` (41), `--curve-N` (41), `--linear-N` (31).

## A conferir antes de usar no artigo

- **Painel (a) da Fig. 1 contra a Fig. 2a do paper original**: temperaturas das transições e valores de S e P.
- **Painel (b)**: os círculos abertos vêm de `selinger_phase_diagram_digitized.csv`, digitalizado a olho de um print
  do diagrama de fases do paper (precisão ~0.02 em T). Antes de publicar, refazer com WebPlotDigitizer sobre o PDF
  original e confirmar o número do painel/figura para citar. A legenda informa o maior desvio entre as curvas.
  Comparação inicial (etapa `phase`, ΔT=0.05): B1=0.09 → T_polar 0.40 (paper ≈0.40); B1=0.30 → 1.05 (≈1.06);
  B1=0.60 → 1.40, transição direta (1.44). Isso também confirma que o B do paper é o B1 do código.
  **Resultado com `phasefine` (2026-10-01):** 30 comparações, desvio máximo 0.04 em T (≈ 2 passos de temperatura,
  incluindo o erro da digitalização); transição direta a partir de B1 = 0.36, como no paper. Faltam os pontos
  polares de B1 = 0.02 e 0.04 (paper: T ≈ 0.12 e 0.20): com B1 pequeno a fase polar é muito deformada por splay e
  P global fica pequeno (B1=0.04: P sobe a 0.18 em T=0.20–0.18 e volta a ~0 enquanto ⟨∇·n⟩ cresce). Nesses casos o
  `make_figures.py` usa como início polar a primeira T em que P > 0.1 (B1=0.04 → 0.20, igual ao paper) e diz isso na legenda.
- As legendas citam a Eq. (12) do rascunho atual.

## Texto do artigo (`latex/`)

- `latex/main.tex` — o rascunho do artigo com a subseção do pear inserida em *Test Problems*, depois de
  *Elastic Behavior* (o mesmo texto está isolado em `latex/pear_section.tex`).
- `latex/pear_refs.bib` — entrada `Dhakal2010`; acrescentar ao `ref.bib` do artigo.
- As figuras são chamadas como `figures/fig_pear_{bulk,droplet,profile}.pdf`: copiar de `figs/` para `figures/`.
- Já corrigido no `main.tex`: definição de $\hat r_{jk}$ após a Eq. do pear e equação do ancoramento como no código,
  $-W_s(\mathbf n\cdot\boldsymbol\beta)^2$.
