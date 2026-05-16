# Research: Embeddings Pre-entrenados vs Entrenados desde Cero

## Contexto
Para clasificación de sentimientos con LSTM/GRU en IMDB, hay dos enfoques principales para la capa de embeddings:
1. **Entrenar desde cero** (random initialization) — los embeddings se aprenden junto con el modelo
2. **Usar embeddings pre-entrenados** (GloVe, Word2Vec, FastText) — inicializar con vectores ya entrenados en corpus grandes

## Hallazgos de la Investigación

### Embeddings Pre-entrenados (GloVe)
- **GloVe-LSTM** logra ~86-90% accuracy en tareas de clasificación de texto ([Medium - Skillcate](https://medium.com/@skillcate/sentiment-classification-using-neural-networks-a-complete-guide-1798aaf357cd), [IJARCCE](https://ijarcce.com/papers/real-world-phishing-and-smishing-detection-using-deep-learning-a-comparative-study-of-lstm-gru-and-glove-embeddings/))
- GloVe 100d o 300d son las dimensiones más comunes
- Ventaja: captura relaciones semánticas aprendidas de corpus masivos (Wikipedia, Common Crawl)
- Desventaja: vectores estáticos, no capturan polisemia ni contexto

### Embeddings desde Cero
- Con suficientes datos (25K reviews de IMDB), los embeddings entrenados desde cero alcanzan ~85-87% accuracy
- Un estudio reciente ([arxiv 2407.12514](https://ar5iv.labs.arxiv.org/html/2407.12514)) encontró que para Transformers, embeddings pre-entrenados de GloVe pueden rendir PEOR que inicialización aleatoria
- Para LSTMs específicamente, GloVe sigue dando ventaja marginal (~1-3% accuracy)
- Un experimento reporta: random embeddings = 75% → GloVe = ~82-85% ([Leyaa AI](https://leyaa.ai/codefly/learn/nlp/part-2/nlp-glove-embeddings/project))

### FastText
- Consistentemente supera a otros embeddings estáticos en datasets con alta variabilidad léxica ([DOAJ](https://doaj.org/article/cd44f37da31446cd8e2c6aee73385dc5))
- Maneja mejor palabras OOV gracias a sub-word information

## Comparación Resumida

| Enfoque | Accuracy típica (IMDB) | Ventajas | Desventajas |
|---------|------------------------|----------|-------------|
| Desde cero (128d) | 85-87% | Simple, embeddings task-specific | Necesita más datos/épocas |
| GloVe 100d | 86-88% | Semántica pre-aprendida, converge más rápido | Archivo grande (~350MB para 100d) |
| GloVe 300d | 87-90% | Mejor representación semántica | Archivo muy grande (~1GB), más parámetros |
| FastText 300d | 87-90% | Maneja OOV, sub-word info | Archivo enorme (~4.5GB) |

## Recomendación

Para este proyecto pedagógico que compara LSTM vs GRU:

**Opción recomendada: Entrenar desde cero (embedding_dim=128)**

Razones:
1. **Foco del proyecto**: La comparación es LSTM vs GRU, no embeddings. Usar embeddings desde cero aísla mejor la variable que queremos estudiar.
2. **Simplicidad**: No requiere descargar archivos grandes de GloVe (~1GB+).
3. **Pedagogía**: Permite entender cómo los embeddings se adaptan a la tarea específica.
4. **Accuracy competitiva**: Con vocab=25K y 20K samples de entrenamiento, la diferencia es marginal (~1-2%).
5. **Reproducibilidad**: Sin dependencia de archivos externos.

**Alternativa si se quiere explorar**: Agregar GloVe como un experimento adicional opcional para comparar el efecto en convergencia y accuracy final.

## Fuentes
- [Sentiment Analysis with LSTM & GloVe - Medium](https://medium.com/@skillcate/sentiment-classification-using-neural-networks-a-complete-guide-1798aaf357cd)
- [On Initializing Transformers with Pre-trained Embeddings - arXiv](https://ar5iv.labs.arxiv.org/html/2407.12514)
- [IJARCCE - LSTM GRU GloVe comparison](https://ijarcce.com/papers/real-world-phishing-and-smishing-detection-using-deep-learning-a-comparative-study-of-lstm-gru-and-glove-embeddings/)
- [DOAJ - Static vs Contextual Embeddings](https://doaj.org/article/cd44f37da31446cd8e2c6aee73385dc5)
