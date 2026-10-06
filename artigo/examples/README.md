# Três exemplos adicionais para o artigo do McLics

```bash
cd artigo/examples
./run.sh            # ou por etapa: twist | cano | bipolar | llbulk
python3 make_figures.py
```

Usa `mclics` do PATH (recompilar com a versão atual: normalização do campo e ancoramentos GHRL unificados).
Roda na GPU (`evol thermalGPU`); `EVOL=thermal ./run.sh` roda na CPU. Rodadas com `DONE` são puladas.
Variáveis: `JOBS` (padrão 2, rodadas simultâneas na mesma GPU), `SEEDS`, `TWIST_P0`, `CANO_NZ`, `BIP_ANCH`, `LL_SIZES`.

| Etapa | Sistema | O que mostra | Rodadas |
|---|---|---|---|
| `twist` | GHRL quiral (E7), 6×6×48, x,y periódicos, z livre; hélice inicial p_i = 0.8, 1.0, 1.25 p0; p0 = 8…40; T = 0.3 | passo = 2π/arcsin(2π/p0) (rede), não p0 | 42 |
| `twistT` | idem, p0 = 8 e 10, T = 0.05…0.5 | passo × T: aproximação da previsão de T = 0 | 20 |
| `cano` | GHRL quiral, p0 = 20, slab 12×12×Nz, `rp` ao longo de x nas duas placas, Nz = 6…46 | degraus de Grandjean | 42 |
| `llbulk` | LL, bulk periódico L = 10…40, resfriamento fino em torno de T_NI | T_NI(L) pelo máximo de C_V | 15 |
| `llheat` | idem, aquecendo a partir do estado alinhado | cerca T_NI junto com o resfriamento (histerese) | 15 |
| `homeo` | LL, gota N = 15, 21, 31, 41, homeotrópico W = 0.03…1 (grade fina em 0.1–0.5) | uniforme/axial → radial; teste de colapso de S_r em W·R (Lavrentovich) | 104 |
| `chiral` | GHRL quiral (E7), gota N = 41, FG W = 1, R/p0 = 0, ±0.1, ±0.2, ±0.3, 0.05, 0.15; 4 seeds | bipolar → bipolar torcida, mão dada pelo sinal de p0 | 36 |

Jobs sugeridos na RTX 4070: `twist`, `twistT`, `cano`: 8; `homeo`, `chiral`: 4; `llbulk`, `llheat`: 2.

## Saída

- `figs/fig_chol.pdf` — (a) passo medido × p0; (b) torção total × d/p (degraus de Grandjean).
- `figs/fig_homeo.pdf` — S_r × W e × W·R; cortes com W fraco e forte.
- `figs/fig_chiral.pdf` — cortes perpendiculares ao eixo bipolar para as duas mãos; torção χ × R/p0.
- `figs/fig_llbulk.pdf` — (a) S × T; (b) C_V × T (resfriamento e aquecimento); (c) T_NI(L) × 1/L³.
- `figures.tex` — legendas em inglês com os números dos dados; `figs/numbers.txt` para conferência.

## Notas

- O passo do GHRL quiral na rede é 2π/arcsin(2π/p0) (ex.: p0 = 10 → 9.25); a diferença para p0 cai como 1/p0².
- Gota bipolar torcida por K22 pequeno (`fg_ghrl`): descartada (tendência invertida). A versão quiral usa só
  |R/p0| ≤ 0.3: com R/p0 = 0.75 (p0 ≈ 13) a gota forma camadas colestéricas quadradas (artefato da rede).
- `homeo`: no teste curto (R = 6.5 e 9.5) o W crítico quase não mudou com R; a campanha completa (R até 19.5,
  grade fina) decide se há colapso em W·R. As legendas do `figures.tex` só valem se os dados confirmarem.
- C_V por sítio = L³·varE/(4T²): o E do `po.dat` conta cada ligação duas vezes.
