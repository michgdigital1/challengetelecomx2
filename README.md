# TelecomX — Parte 2: Predicción de Churn
**Challenge Data Science**

---

## 1. Propósito del Análisis

El objetivo principal es **predecir qué clientes tienen mayor probabilidad de cancelar su servicio (churn)** en TelecomX, usando modelos de clasificación supervisada.

Con este análisis, la empresa puede anticiparse al problema, identificar clientes en riesgo y activar estrategias de retención antes de que cancelen.

---

## 2. Estructura del Proyecto

```
telecomx-parte2/
│
├── TelecomX_parte2_LATAM.ipynb   # Notebook principal con todo el pipeline
├── TelecomX_Data.json            # Dataset tratado (resultado de la Parte 1)
└── README.md                     # Este archivo
```

> El notebook carga el dataset directamente desde GitHub al ejecutarse. No es necesario descargar el JSON por separado.

---

## 3. Proceso de Preparación de los Datos

### Variables del dataset

| Tipo | Ejemplos |
|------|----------|
| **Numéricas** | `tenure`, `MonthlyCharges`, `TotalCharges` |
| **Categóricas** | `Contract`, `InternetService`, `PaymentMethod`, `gender`, etc. |
| **Objetivo** | `Churn` → codificada como `0` (No) / `1` (Sí) |

### Etapas aplicadas

**Limpieza**
- `TotalCharges` convertida a numérico (contenía espacios en blanco)
- Valores nulos imputados con la mediana
- Columna `customerID` eliminada (no aporta valor predictivo)

**Codificación**
- Variables categóricas → **One-Hot Encoding** con `pd.get_dummies(drop_first=True)`
- Variable objetivo → mapeada a binario: `Yes → 1`, `No → 0`

**Balanceo de clases**
- El dataset original presenta ~74% No Churn / ~26% Churn
- Se aplicó **oversampling** de la clase minoritaria con `sklearn.utils.resample`
- Resultado: dataset balanceado 50/50

**Normalización**
- Se usa `StandardScaler` **solo para modelos sensibles a escala**:
  - ✅ Regresión Logística y KNN → usan `X_train_scaled`
  - ❌ Árbol de Decisión y Random Forest → usan `X_train` sin escalar

**División train/test**
- Split **80% entrenamiento / 20% prueba**
- Estratificado para preservar la proporción de clases en ambos conjuntos

---

## 4. Insights del Análisis Exploratorio (EDA)

**Distribución de Churn**
- ~26% de los clientes cancelaron el servicio
- Existe desbalance de clases que requiere tratamiento

**Variables con mayor correlación con Churn**
- 🔴 `Contract_Month-to-month` — clientes sin contrato fijo cancelan 3–4x más
- 🔴 `InternetService_Fiber optic` — mayor tasa de churn, posiblemente por precio/expectativas
- 🔴 `PaymentMethod_Electronic check` — menor compromiso, mayor cancelación
- 🔵 `tenure` — a mayor antigüedad, menor probabilidad de churn
- 🔵 `Contract_Two year` — el contrato largo reduce drásticamente la cancelación

**Boxplots clave**
- Clientes que cancelan tienen **menor antigüedad promedio** (~18 meses vs ~37 meses)
- Los que cancelan pagan un **cargo mensual más alto** en promedio

---

## 5. Instrucciones para Ejecutar el Notebook

### Requisitos

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

> El notebook fue desarrollado con Python 3.10+.

### Carga de datos

El notebook intenta cargar el JSON automáticamente desde GitHub:

```python
URL = "https://raw.githubusercontent.com/michgdigital1/challengetelecomx2/refs/heads/main/TelecomX_Data_Part2.json"
```

Si no hay conexión a internet, se activa un **dataset de respaldo** generado con las mismas características estadísticas, por lo que el notebook siempre puede ejecutarse completo.

### Pasos para ejecutar

1. Abrir `TelecomX_parte2_LATAM.ipynb` en **Google Colab** 
2. Ejecutar las celdas en orden (Ejecutar todo / Run All)
3. No se requiere configuración adicional

---

## 6. Modelos Entrenados

| Modelo | Normalización | Métricas principales |
|--------|--------------|----------------------|
| Regresión Logística | ✅ Sí | Interpretable, buen baseline |
| KNN (k=7) | ✅ Sí | Basado en distancias |
| Árbol de Decisión | ❌ No | Fácil de visualizar |
| Random Forest | ❌ No | Mejor rendimiento general |

Métricas evaluadas: **Accuracy, Precision, Recall, F1-Score, ROC-AUC**, matrices de confusión y validación cruzada (5-fold).

---

*TelecomX · Data Science*
