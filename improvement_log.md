# Log de Mejoras para Precisión >90% en NER

## Objetivo
- **Precisión**: >90%
- **Recall**: Mantener similar al actual (~100%)
- **F1**: >90%

## Estado Actual (--limit 5)
- **Precisión**: 33.33%
- **Recall**: 100%
- **F1**: 50%
- **Predictions**: 30
- **Gold**: 10
- **TP**: 10

## Análisis del Problema
- **Recall perfecto**: El sistema encuentra todas las entidades gold
- **Precisión baja**: Muchos falsos positivos (20 FP vs 10 TP)
- **Causa principal**: Umbrales muy relajados para pruebas cortas

## Plan de Mejoras

### Fase 1: Ajuste de Umbrales y Filtros
- [ ] Aumentar `min_occurrences` de 1 a 2-3
- [ ] Aumentar `reliability_thresh` de 0.10 a 0.50-0.70
- [ ] Implementar filtro de confianza por entidad
- [ ] Añadir validación cruzada entre chunks

### Fase 2: Mejoras en el Prompt y LLM
- [ ] Probar con `qwen3:8b` (modelo más potente)
- [ ] Mejorar el prompt para ser más estricto
- [ ] Implementar validación de evidencia más rigurosa
- [ ] Añadir ejemplos negativos en el prompt

### Fase 3: Post-procesamiento y Validación
- [ ] Implementar filtro de coherencia semántica
- [ ] Validación de tipos de entidad
- [ ] Filtro de frecuencia global
- [ ] Eliminación de duplicados y variantes

### Fase 4: Optimización del Pipeline
- [ ] Ajustar tamaño de chunks y overlap
- [ ] Optimizar batch size
- [ ] Implementar early stopping para entidades de baja confianza

## Iteraciones

### Iteración 1: Ajuste de Umbrales Básicos
**Fecha**: 2025-01-27
**Cambios**:
- Aumentar `min_occurrences` de 1 a 2
- Aumentar `reliability_thresh` de 0.10 a 0.50
- Activar `self_consistency` por defecto
- Implementar filtro de confianza más estricto (2+ exact matches o 1 exact + thresholds)
- Mejorar prompt para ser más conservador
- Añadir filtro de múltiples chunks para entidades sin exact matches
- Añadir ejemplos negativos en el prompt

**Resultados Esperados**: Precisión 60-70%, Recall 90-95%

**Resultados Obtenidos (llama3.2:3b)**:
- **Precisión**: 50.00% (mejoró de 33.33%)
- **Recall**: 80.00% (bajó de 100%)
- **F1**: 61.54% (mejoró de 50%)
- **Predictions**: 16 (redujo de 30)
- **Gold**: 10
- **TP**: 8 (bajó de 10)

**Análisis**:
- ✅ **Precisión mejoró**: De 33.33% a 50.00% (+16.67 puntos)
- ❌ **Recall bajó**: De 100% a 80.00% (-20 puntos)
- ✅ **F1 mejoró**: De 50% a 61.54% (+11.54 puntos)
- ✅ **Falsos positivos reducidos**: De 20 a 8 (-12 FP)

**Observaciones**:
- El filtro más estricto está funcionando para reducir FPs
- Algunas entidades gold se están perdiendo por ser demasiado estrictos
- Necesitamos ajustar el balance entre precisión y recall

---

## Notas de Implementación
- Cada iteración debe probarse con `--limit 5` primero
- Documentar métricas exactas, no inventar números
- Probar tanto `llama3.2:3b` como `qwen3:8b`
- Mantener registro de tiempo y recursos utilizados
