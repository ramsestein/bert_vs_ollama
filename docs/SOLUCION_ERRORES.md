## 🐛 Solución de Problemas

### Ollama No Responde

```bash
# Verificar estado
ollama list

# Reiniciar servicio
ollama serve
```

### Errores de Memoria

- Reducir `--limit` para procesar menos documentos
- Verificar que Ollama tenga suficiente RAM disponible
- Usar estrategias con chunks más pequeños

### Baja Precisión

- Ajustar `--confidence_threshold`
- Verificar que los modelos estén descargados
- Revisar logs de debug para errores específicos

## 📚 Archivos de Configuración

### Formato de Entrada de texto a procesar(JSONL)

⚠️ **CRÍTICO**: El campo "Entidad" define **EXACTAMENTE** qué entidades se extraerán del texto. El sistema NO detectará entidades que no estén en esta lista.
