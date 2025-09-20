# RefMaster - Reference Matching (EQ Curve + LUFS)

RefMaster aprende uma curva de EQ a partir de uma faixa de referência e a aplica na sua mix, além de alinhar loudness para um alvo LUFS. Gera relatório com curva aprendida e estatísticas de loudness.

## Funcionalidades
- Medição de loudness integrada (LUFS) e normalização para alvo.
- EQ matching por banda usando espectro médio e suavização.
- Exporta áudio processado e relatório HTML com gráficos.
- CLI simples e reprodutível.

## Instalação
```bash
pip install -r requirements.txt
```

## Uso
```bash
# medir loudness e espectro
python src/refmaster.py analyze --track audio/meu_mix.wav --report

# casar com referência e normalizar para -14 LUFS
python src/refmaster.py match --track audio/meu_mix.wav --reference audio/ref.wav --lufs -14 --out out_matched.wav --report
```

## Limitações
EQ matching é estático; variações por seção podem exigir automação manual. Use o relatório para validar.

## Licença
MIT.
