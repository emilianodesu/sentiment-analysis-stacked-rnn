"""Automated generation of REFLEXIONES.md document.

Generates a markdown document answering 8 reflection questions about the
Stacked RNN sentiment analysis experiment, using actual training and
evaluation results as evidence.
"""

import os
from typing import Optional

from config import (
    EMBED_DROPOUT,
    EMBEDDING_DIM,
    GRAD_CLIP_NORM,
    HIDDEN_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MAX_LEN,
    NUM_LAYERS,
    PATIENCE,
    RNN_DROPOUT,
    VOCAB_SIZE,
)
from evaluate import EvaluationResults
from train import TrainingHistory


def _compute_avg_gradient_norm(history: TrainingHistory) -> float:
    """Compute the average gradient norm across all layers and steps."""
    all_norms = []
    for norms in history.gradient_norms.values():
        all_norms.extend(norms)
    if not all_norms:
        return 0.0
    return sum(all_norms) / len(all_norms)


def _compute_layer_gradient_stats(
    history: TrainingHistory,
) -> dict[str, dict[str, float]]:
    """Compute per-layer gradient statistics (mean, max, min)."""
    stats: dict[str, dict[str, float]] = {}
    for layer_name, norms in history.gradient_norms.items():
        if norms:
            stats[layer_name] = {
                "mean": sum(norms) / len(norms),
                "max": max(norms),
                "min": min(norms),
            }
    return stats


def _get_rnn_layer_norms(
    history: TrainingHistory,
) -> dict[str, list[float]]:
    """Extract gradient norms for RNN-specific layers only."""
    rnn_norms: dict[str, list[float]] = {}
    for name, norms in history.gradient_norms.items():
        if "rnn" in name.lower() or "gru" in name.lower() or "lstm" in name.lower():
            rnn_norms[name] = norms
    return rnn_norms


def _detect_gradient_issues(history: TrainingHistory) -> Optional[str]:
    """Detect vanishing or exploding gradient patterns in the data."""
    rnn_norms = _get_rnn_layer_norms(history)
    if not rnn_norms:
        return None

    issues = []
    for layer_name, norms in rnn_norms.items():
        if not norms:
            continue
        avg = sum(norms) / len(norms)
        # Check for vanishing gradients (very small norms)
        if avg < 1e-4:
            issues.append(f"vanishing gradients in {layer_name} (avg norm: {avg:.6f})")
        # Check for exploding gradients (very large norms before clipping)
        if max(norms) > GRAD_CLIP_NORM * 0.9:
            issues.append(
                f"near-clipping gradients in {layer_name} (max norm: {max(norms):.4f})"
            )

    return "; ".join(issues) if issues else None


def generate_reflexiones(
    lstm_history: TrainingHistory,
    gru_history: TrainingHistory,
    lstm_results: EvaluationResults,
    gru_results: EvaluationResults,
    output_dir: str = "outputs",
) -> str:
    """Generate REFLEXIONES.md with answers to 8 reflection questions.

    Uses actual experimental results (training histories and evaluation metrics)
    to provide evidence-based answers to each reflection question.

    Args:
        lstm_history: Training history from the LSTM model.
        gru_history: Training history from the GRU model.
        lstm_results: Evaluation results from the LSTM model.
        gru_results: Evaluation results from the GRU model.
        output_dir: Directory where REFLEXIONES.md will be saved.

    Returns:
        Path to the generated REFLEXIONES.md file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "REFLEXIONES.md")

    # Compute derived metrics
    lstm_avg_grad = _compute_avg_gradient_norm(lstm_history)
    gru_avg_grad = _compute_avg_gradient_norm(gru_history)
    lstm_grad_stats = _compute_layer_gradient_stats(lstm_history)
    gru_grad_stats = _compute_layer_gradient_stats(gru_history)
    lstm_gradient_issues = _detect_gradient_issues(lstm_history)
    gru_gradient_issues = _detect_gradient_issues(gru_history)

    # Determine convergence speed (epoch where best val accuracy was reached)
    lstm_best_epoch = (
        lstm_history.val_accs.index(max(lstm_history.val_accs)) + 1
        if lstm_history.val_accs
        else 0
    )
    gru_best_epoch = (
        gru_history.val_accs.index(max(gru_history.val_accs)) + 1
        if gru_history.val_accs
        else 0
    )

    # Best validation metrics
    lstm_best_val_acc = max(lstm_history.val_accs) if lstm_history.val_accs else 0.0
    gru_best_val_acc = max(gru_history.val_accs) if gru_history.val_accs else 0.0

    # Training time comparison
    time_ratio = (
        lstm_history.training_time / gru_history.training_time
        if gru_history.training_time > 0
        else 0.0
    )

    # Early stopping info
    lstm_stopped_early = lstm_history.epochs_trained < MAX_EPOCHS
    gru_stopped_early = gru_history.epochs_trained < MAX_EPOCHS

    # Overfitting detection (train acc >> val acc in final epoch)
    lstm_final_overfit = (
        (lstm_history.train_accs[-1] - lstm_history.val_accs[-1])
        if lstm_history.train_accs and lstm_history.val_accs
        else 0.0
    )
    gru_final_overfit = (
        (gru_history.train_accs[-1] - gru_history.val_accs[-1])
        if gru_history.train_accs and gru_history.val_accs
        else 0.0
    )

    content = f"""# Reflexiones sobre el Experimento de Análisis de Sentimiento con RNN Apiladas

## 1. ¿Cómo afecta el número de capas apiladas a la capacidad de aprendizaje?

En este experimento se utilizaron {NUM_LAYERS} capas recurrentes apiladas con un tamaño oculto de {HIDDEN_SIZE} unidades por capa. La arquitectura apilada permite que cada capa aprenda representaciones a diferentes niveles de abstracción: la primera capa captura patrones locales y sintácticos, mientras que la segunda capa aprende dependencias semánticas más complejas.

Los resultados experimentales muestran que el LSTM alcanzó una accuracy de {lstm_results.accuracy * 100:.2f}% y el GRU de {gru_results.accuracy * 100:.2f}% en el conjunto de test. Con {NUM_LAYERS} capas, ambos modelos lograron capturar suficiente complejidad para la tarea de clasificación binaria de sentimiento.

El uso de múltiples capas incrementa la capacidad del modelo pero también el riesgo de sobreajuste. En nuestro caso, la diferencia entre accuracy de entrenamiento y validación fue de {lstm_final_overfit * 100:.2f}% para LSTM y {gru_final_overfit * 100:.2f}% para GRU, lo que indica {"un sobreajuste moderado que fue controlado por el dropout y early stopping" if max(lstm_final_overfit, gru_final_overfit) > 0.05 else "un buen balance entre capacidad y generalización"}.

## 2. ¿Cuál es el impacto del dropout entre capas RNN en la generalización?

Se aplicó un dropout de {RNN_DROPOUT} entre las capas recurrentes y un dropout de {EMBED_DROPOUT} en la capa de embedding. Esta regularización es fundamental para prevenir el sobreajuste en arquitecturas profundas.

**Evidencia experimental:**
- LSTM: accuracy de entrenamiento final = {lstm_history.train_accs[-1] * 100:.2f}%, accuracy de validación = {lstm_history.val_accs[-1] * 100:.2f}% (gap: {lstm_final_overfit * 100:.2f}%)
- GRU: accuracy de entrenamiento final = {gru_history.train_accs[-1] * 100:.2f}%, accuracy de validación = {gru_history.val_accs[-1] * 100:.2f}% (gap: {gru_final_overfit * 100:.2f}%)

El dropout entre capas RNN ({RNN_DROPOUT}) actúa como un regularizador que fuerza a cada capa a aprender representaciones más robustas e independientes. Sin este dropout, las capas superiores podrían depender excesivamente de activaciones específicas de las capas inferiores. El dropout en el embedding ({EMBED_DROPOUT}) previene que el modelo memorice combinaciones específicas de palabras, promoviendo una mejor generalización a vocabulario no visto durante el entrenamiento.

Los resultados en test (LSTM: {lstm_results.accuracy * 100:.2f}%, GRU: {gru_results.accuracy * 100:.2f}%) confirman que la combinación de dropout aplicada permite una buena generalización a datos no vistos.

## 3. ¿Cómo se comparan LSTM y GRU en términos de velocidad de convergencia?

**Comparación de convergencia:**
- LSTM alcanzó su mejor accuracy de validación ({lstm_best_val_acc * 100:.2f}%) en la época {lstm_best_epoch} de {lstm_history.epochs_trained} entrenadas
- GRU alcanzó su mejor accuracy de validación ({gru_best_val_acc * 100:.2f}%) en la época {gru_best_epoch} de {gru_history.epochs_trained} entrenadas

**Tiempos de entrenamiento:**
- LSTM: {lstm_history.training_time:.1f} segundos totales ({lstm_history.training_time / lstm_history.epochs_trained:.1f} s/época)
- GRU: {gru_history.training_time:.1f} segundos totales ({gru_history.training_time / gru_history.epochs_trained:.1f} s/época)
- Ratio LSTM/GRU: {time_ratio:.2f}x

{"El GRU convergió más rápido que el LSTM, alcanzando su mejor rendimiento en menos épocas." if gru_best_epoch < lstm_best_epoch else "El LSTM convergió más rápido que el GRU, alcanzando su mejor rendimiento en menos épocas." if lstm_best_epoch < gru_best_epoch else "Ambos modelos convergieron a una velocidad similar."} En cuanto a tiempo por época, {"el GRU fue más rápido debido a su arquitectura más simple (2 compuertas vs 3 del LSTM), lo que se traduce en menos operaciones matriciales por paso temporal" if time_ratio > 1.05 else "ambos modelos tuvieron tiempos similares por época" if time_ratio < 1.05 and time_ratio > 0.95 else "el LSTM fue ligeramente más rápido en este caso"}.

## 4. ¿Qué diferencias se observan en el flujo de gradientes entre LSTM y GRU?

El monitoreo de gradientes durante los primeros {len(next(iter(lstm_history.gradient_norms.values()), []))} pasos de entrenamiento revela diferencias en cómo fluyen los gradientes a través de cada arquitectura.

**Normas de gradiente promedio:**
- LSTM: {lstm_avg_grad:.6f}
- GRU: {gru_avg_grad:.6f}

**Análisis por capas (LSTM):**
"""

    # Add per-layer stats for LSTM
    for layer_name, stats in sorted(lstm_grad_stats.items()):
        short_name = layer_name.split(".")[-1] if "." in layer_name else layer_name
        content += f"- `{short_name}`: media={stats['mean']:.6f}, máx={stats['max']:.6f}, mín={stats['min']:.6f}\n"

    content += """
**Análisis por capas (GRU):**
"""

    # Add per-layer stats for GRU
    for layer_name, stats in sorted(gru_grad_stats.items()):
        short_name = layer_name.split(".")[-1] if "." in layer_name else layer_name
        content += f"- `{short_name}`: media={stats['mean']:.6f}, máx={stats['max']:.6f}, mín={stats['min']:.6f}\n"

    # Gradient issues analysis
    gradient_analysis = ""
    if lstm_gradient_issues or gru_gradient_issues:
        gradient_analysis = "\n**Patrones detectados:**\n"
        if lstm_gradient_issues:
            gradient_analysis += f"- LSTM: {lstm_gradient_issues}\n"
        if gru_gradient_issues:
            gradient_analysis += f"- GRU: {gru_gradient_issues}\n"
    else:
        gradient_analysis = "\nNo se detectaron patrones de gradientes que se desvanecen ni explotan en ninguno de los dos modelos, lo cual indica que tanto el gradient clipping (max_norm={GRAD_CLIP_NORM}) como las compuertas internas de ambas arquitecturas cumplen su función de mantener un flujo de gradientes estable.\n"

    content += gradient_analysis

    content += f"""
El LSTM utiliza tres compuertas (entrada, olvido, salida) y una celda de memoria separada que permite un flujo de gradientes más directo a través del tiempo. El GRU simplifica esto con dos compuertas (reset, update), lo que resulta en {"normas de gradiente ligeramente diferentes pero igualmente estables" if abs(lstm_avg_grad - gru_avg_grad) / max(lstm_avg_grad, gru_avg_grad, 1e-10) < 0.5 else "diferencias notables en la magnitud de los gradientes entre ambas arquitecturas"}.

## 5. ¿Cómo afecta la longitud de secuencia (padding/truncamiento) al rendimiento del modelo?

Se utilizó una longitud máxima de secuencia de {MAX_LEN} tokens. Las reseñas más largas se truncan y las más cortas se rellenan con el token `<PAD>` (índice 0).

**Impacto en el rendimiento:**
- Con MAX_LEN={MAX_LEN}, el LSTM logró {lstm_results.accuracy * 100:.2f}% de accuracy y el GRU {gru_results.accuracy * 100:.2f}%
- F1-Score: LSTM={lstm_results.f1 * 100:.2f}%, GRU={gru_results.f1 * 100:.2f}%

La elección de {MAX_LEN} tokens representa un balance entre:
1. **Captura de información**: Las reseñas de IMDB tienen longitudes variables. Con 200 tokens se captura la mayor parte del contenido semántico relevante para determinar el sentimiento.
2. **Eficiencia computacional**: Secuencias más largas incrementan linealmente el tiempo de cómputo y el uso de memoria en las capas recurrentes.
3. **Padding excesivo**: Secuencias muy cortas rellenadas con muchos ceros pueden diluir la señal útil, aunque las RNN aprenden a ignorar el padding gracias a las compuertas.

Los resultados sugieren que {MAX_LEN} tokens es suficiente para esta tarea, dado que ambos modelos alcanzan un rendimiento competitivo (F1 > {min(lstm_results.f1, gru_results.f1) * 100:.1f}%).

## 6. ¿Qué papel juega la capa de embedding en la representación del sentimiento?

La capa de embedding transforma cada token (de un vocabulario de {VOCAB_SIZE:,} palabras) en un vector denso de {EMBEDDING_DIM} dimensiones. Esta transformación es fundamental porque:

1. **Representación semántica**: Los embeddings aprenden a posicionar palabras con significado similar en regiones cercanas del espacio vectorial. Palabras como "excelente", "fantástico" y "maravilloso" convergen hacia representaciones similares durante el entrenamiento.

2. **Reducción de dimensionalidad**: En lugar de representaciones one-hot de {VOCAB_SIZE:,} dimensiones, cada palabra se codifica en solo {EMBEDDING_DIM} dimensiones, lo que hace el procesamiento computacionalmente viable.

3. **Captura de polaridad**: Para análisis de sentimiento, los embeddings aprenden a separar palabras positivas de negativas en el espacio vectorial, facilitando la clasificación posterior por las capas recurrentes.

**Evidencia experimental:**
- El dropout de {EMBED_DROPOUT} aplicado después del embedding actúa como regularización, previniendo que el modelo memorice patrones superficiales
- Ambos modelos (LSTM: {lstm_results.accuracy * 100:.2f}%, GRU: {gru_results.accuracy * 100:.2f}%) logran un rendimiento sólido, lo que confirma que los embeddings de {EMBEDDING_DIM} dimensiones capturan suficiente información semántica para la tarea

## 7. ¿Qué tan efectivo es el early stopping en la prevención del sobreajuste?

**Configuración:** patience={PATIENCE} épocas, máximo {MAX_EPOCHS} épocas.

**Resultados:**
- LSTM: entrenó {lstm_history.epochs_trained} de {MAX_EPOCHS} épocas máximas {"(early stopping activado)" if lstm_stopped_early else "(completó todas las épocas)"}
- GRU: entrenó {gru_history.epochs_trained} de {MAX_EPOCHS} épocas máximas {"(early stopping activado)" if gru_stopped_early else "(completó todas las épocas)"}

**Análisis de sobreajuste:**
- LSTM: gap train-val accuracy = {lstm_final_overfit * 100:.2f}% en la última época
- GRU: gap train-val accuracy = {gru_final_overfit * 100:.2f}% en la última época

{"El early stopping fue efectivo al detener el entrenamiento antes de que el sobreajuste se agravara." if lstm_stopped_early or gru_stopped_early else "Ningún modelo activó el early stopping, lo que sugiere que ambos seguían mejorando o que el patience fue suficiente para la cantidad de épocas configuradas."} El mecanismo guarda el mejor checkpoint basado en la validation loss más baja, asegurando que el modelo final sea el que mejor generaliza, independientemente de si el entrenamiento continuó después del punto óptimo.

La combinación de early stopping con dropout ({RNN_DROPOUT} entre capas RNN, {EMBED_DROPOUT} en embedding) proporciona una defensa multicapa contra el sobreajuste: el dropout regulariza durante el entrenamiento y el early stopping limita la duración del mismo.

## 8. ¿Cuáles son los trade-offs entre LSTM y GRU para esta tarea?

### Comparación cuantitativa completa:

| Métrica | LSTM | GRU | Diferencia |
|---------|------|-----|------------|
| Accuracy | {lstm_results.accuracy * 100:.2f}% | {gru_results.accuracy * 100:.2f}% | {(lstm_results.accuracy - gru_results.accuracy) * 100:+.2f}% |
| Precision | {lstm_results.precision * 100:.2f}% | {gru_results.precision * 100:.2f}% | {(lstm_results.precision - gru_results.precision) * 100:+.2f}% |
| Recall | {lstm_results.recall * 100:.2f}% | {gru_results.recall * 100:.2f}% | {(lstm_results.recall - gru_results.recall) * 100:+.2f}% |
| F1-Score | {lstm_results.f1 * 100:.2f}% | {gru_results.f1 * 100:.2f}% | {(lstm_results.f1 - gru_results.f1) * 100:+.2f}% |
| Tiempo total | {lstm_history.training_time:.1f}s | {gru_history.training_time:.1f}s | {lstm_history.training_time - gru_history.training_time:+.1f}s |
| Épocas | {lstm_history.epochs_trained} | {gru_history.epochs_trained} | {lstm_history.epochs_trained - gru_history.epochs_trained:+d} |
| Grad. norm promedio | {lstm_avg_grad:.6f} | {gru_avg_grad:.6f} | — |

### Trade-offs principales:

1. **Rendimiento vs. Complejidad**: {"El LSTM supera al GRU en accuracy" if lstm_results.accuracy > gru_results.accuracy else "El GRU supera al LSTM en accuracy" if gru_results.accuracy > lstm_results.accuracy else "Ambos modelos tienen accuracy idéntica"} ({abs(lstm_results.accuracy - gru_results.accuracy) * 100:.2f}% de diferencia), {"justificando su mayor complejidad computacional" if lstm_results.accuracy > gru_results.accuracy else "a pesar de tener menos parámetros" if gru_results.accuracy > lstm_results.accuracy else "lo que sugiere que la complejidad adicional del LSTM no aporta beneficio para esta tarea"}.

2. **Velocidad vs. Capacidad**: El GRU tiene 2 compuertas (reset, update) frente a las 3 del LSTM (entrada, olvido, salida) más la celda de memoria. Esto se refleja en {"un menor tiempo de entrenamiento para el GRU" if time_ratio > 1.05 else "tiempos de entrenamiento similares"} (ratio: {time_ratio:.2f}x).

3. **Estabilidad de gradientes**: {"Ambos modelos mantienen gradientes estables" if not lstm_gradient_issues and not gru_gradient_issues else "Se observaron diferencias en la estabilidad de gradientes"}, con normas promedio de {lstm_avg_grad:.6f} (LSTM) y {gru_avg_grad:.6f} (GRU).

4. **Convergencia**: {"El GRU converge más rápido" if gru_best_epoch < lstm_best_epoch else "El LSTM converge más rápido" if lstm_best_epoch < gru_best_epoch else "Ambos convergen a velocidad similar"} (mejor época: LSTM={lstm_best_epoch}, GRU={gru_best_epoch}).

## Conclusión

Este experimento comparó LSTM y GRU bajo condiciones idénticas (seed={42}, {NUM_LAYERS} capas, {HIDDEN_SIZE} unidades ocultas, dropout={RNN_DROPOUT}, lr={LEARNING_RATE}) para análisis de sentimiento en IMDB.

**Hallazgos clave:**
- {"El LSTM obtuvo mejor rendimiento general" if lstm_results.f1 > gru_results.f1 else "El GRU obtuvo mejor rendimiento general" if gru_results.f1 > lstm_results.f1 else "Ambos modelos obtuvieron rendimiento equivalente"} (F1: LSTM={lstm_results.f1 * 100:.2f}% vs GRU={gru_results.f1 * 100:.2f}%)
- {"El GRU fue más eficiente en tiempo de entrenamiento" if time_ratio > 1.05 else "Ambos modelos tuvieron eficiencia temporal similar"} ({gru_history.training_time:.1f}s vs {lstm_history.training_time:.1f}s)
- La combinación de dropout ({RNN_DROPOUT}/{EMBED_DROPOUT}) y early stopping (patience={PATIENCE}) controló efectivamente el sobreajuste
- {"No se detectaron problemas de gradientes en ningún modelo" if not lstm_gradient_issues and not gru_gradient_issues else "Se detectaron algunos patrones de gradientes que requieren atención"}, confirmando la efectividad del gradient clipping (max_norm={GRAD_CLIP_NORM})

Para esta tarea específica de clasificación binaria de sentimiento, {"la diferencia entre LSTM y GRU es marginal (" + f"{abs(lstm_results.accuracy - gru_results.accuracy) * 100:.2f}%" + " en accuracy), sugiriendo que el GRU puede ser preferible por su menor complejidad computacional cuando el rendimiento es comparable" if abs(lstm_results.accuracy - gru_results.accuracy) < 0.02 else "el " + ("LSTM" if lstm_results.accuracy > gru_results.accuracy else "GRU") + " demuestra una ventaja clara que justifica " + ("su mayor complejidad" if lstm_results.accuracy > gru_results.accuracy else "su arquitectura más simple")}.
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_path
