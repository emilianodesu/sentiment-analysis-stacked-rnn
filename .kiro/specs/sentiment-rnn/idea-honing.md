# Idea Honing

## Q1: ¿Qué tipo de capa recurrente prefieres usar: LSTM o GRU? El rough idea menciona ambas como opciones. ¿Tienes preferencia, o te gustaría implementar ambas y comparar resultados?
**A:** Implementar ambas y compararlas.

## Q2: Para la comparación LSTM vs GRU, ¿quieres que ambos modelos compartan exactamente los mismos hiperparámetros (hidden size, num_layers, dropout, etc.) para que la comparación sea justa, o prefieres optimizar cada uno por separado?
**A:** Mismos hiperparámetros para una comparación justa.

## Q3: ¿Qué tamaño máximo de vocabulario tienes en mente? Valores comunes para IMDB van de 20,000 a 50,000 palabras. ¿Tienes alguna preferencia, o quieres usar un valor estándar como 25,000?
**A:** 25,000 con min_freq=2 (basado en investigación documentada en research/vocabulary-and-hyperparameters.md).

## Q4: Para la longitud máxima de secuencia (max_len), la investigación muestra que la mediana de palabras por review es ~174. ¿Prefieres usar 200 (cubre la mayoría) o 256 (más holgura para reviews largas)?
**A:** 200.

## Q5: Para el split de datos, el dataset IMDB viene con 25K train y 25K test. ¿Quieres crear un subset de validación a partir del train set (por ejemplo, 80/20 → 20K train, 5K val), o prefieres otro esquema de partición?
**A:** 80/20 — 20K train, 5K val, 25K test.

## Q6: Para la estructura del código, ¿prefieres un solo notebook/script monolítico o una estructura modular con archivos separados (e.g., data.py, model.py, train.py, evaluate.py, visualize.py)?
**A:** Estructura modular con archivos separados.

## Q7: ¿Quieres que el proyecto soporte ejecución en GPU automáticamente (detectar CUDA/MPS y mover tensores al dispositivo disponible), o está bien que corra solo en CPU?
**A:** Sí, soporte automático de GPU (CUDA/MPS).

## Q8: Para las preguntas de reflexión mencionadas en el rough idea (impacto del padding, LSTM vs GRU, vanishing gradients, etc.), ¿quieres que se generen como un documento markdown separado con las respuestas basadas en los resultados experimentales, o prefieres responderlas tú mismo como ejercicio?
**A:** Documento markdown separado con respuestas generadas basadas en resultados experimentales.

## Q9: ¿Cuántas épocas de entrenamiento quieres usar? La investigación sugiere que para IMDB el overfitting suele empezar entre la época 5-7. ¿Te parece bien usar 10 épocas con early stopping basado en validation loss, o prefieres un número fijo?
**A:** 10 épocas con early stopping basado en validation loss.

## Q10: ¿Hay algún aspecto adicional que quieras cubrir o alguna restricción que no hayamos discutido? Por ejemplo: ¿usar embeddings pre-entrenados (GloVe/Word2Vec) o entrenar embeddings desde cero? ¿Algún requisito de reproducibilidad (seeds fijos)?
**A:** Seed fijo = 42 para reproducibilidad. Embeddings entrenados desde cero (embedding_dim=128), basado en investigación documentada en research/embeddings-pretrained-vs-scratch.md.
