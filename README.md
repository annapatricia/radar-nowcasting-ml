## Nowcasting (Previsão de curto prazo)

Modelo supervisionado que recebe 5 frames consecutivos de radar e prevê
se haverá célula convectiva intensa no próximo frame.

### Resultados (validação)

Confusion Matrix:

[[ 31  14]
 [  3 152]]

- Alta sensibilidade para eventos intensos (recall ~ 0.98)
- Alguns falsos positivos (modelo conservador)

### Exemplos

Veja em:
outputs/figures/future_examples/
- TP.png
- TN.png
- FP.png
- FN.png

## 🖼 Exemplos de Previsão (Nowcasting)

### True Positive (acerto evento intenso)
![TP](outputs/figures/future_examples/TP.png)

### True Negative (acerto ausência de evento)
![TN](outputs/figures/future_examples/TN.png)

### False Positive (alarme falso)
![FP](outputs/figures/future_examples/FP.png)

### False Negative (evento perdido)
![FN](outputs/figures/future_examples/FN.png)
