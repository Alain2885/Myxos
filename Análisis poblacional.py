import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scikit_posthocs import posthoc_dunn
import pandas as pd

# ------------------ Funciones de procesamiento de imágenes ------------------
def detectar_barra_escala(img_gray, img_color, longitud_real_um):
    _, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 3))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contornos, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    barra = None
    max_w = 0
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if aspect_ratio > 10 and w > max_w:
            barra = (x, y, w, h)
            max_w = w
    if barra is None:
        return None, None, None
    x, y, w, h = barra
    pix_por_um = w / longitud_real_um
    cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return pix_por_um, barra, img_color

def binarizar_esporas(img_gray):
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    return binary

def feret_diameter(contour):
    hull = cv2.convexHull(contour)
    hull = hull.squeeze()
    if len(hull) < 2:
        return 0.0
    max_dist = 0
    for i in range(len(hull)):
        for j in range(i+1, len(hull)):
            dist = np.linalg.norm(hull[i] - hull[j])
            if dist > max_dist:
                max_dist = dist
    return max_dist

def identificar_esporas(binary, img_gray_original, pix_um, img_color,
                        min_area_px=20, circularidad_min=0.6, margin=5):
    h, w = binary.shape
    inv = cv2.bitwise_not(binary)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=1)
    contornos, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stats = []
    idx = 1

    for cnt in contornos:
        if any(p[0][0] < margin or p[0][0] >= w - margin or p[0][1] < margin or p[0][1] >= h - margin for p in cnt):
            continue
        area_px = cv2.contourArea(cnt)
        if area_px < min_area_px:
            continue
        perim_px = cv2.arcLength(cnt, True)
        if perim_px == 0:
            continue
        circularidad = 4 * np.pi * area_px / (perim_px * perim_px)
        if circularidad < circularidad_min:
            continue

        diam_eq_px = 2 * np.sqrt(area_px / np.pi)
        feret_px = feret_diameter(cnt)
        area_um2 = area_px / (pix_um * pix_um)
        perim_um = perim_px / pix_um
        diam_eq_um = diam_eq_px / pix_um
        feret_um = feret_px / pix_um

        (cx, cy), radio = cv2.minEnclosingCircle(cnt)
        centro = (int(cx), int(cy))
        radio = int(radio)

        cv2.drawContours(img_color, [cnt], -1, (0, 255, 0), 2)
        cv2.circle(img_color, centro, radio, (255, 0, 0), 1)
        cv2.putText(img_color, str(idx), (centro[0]-10, centro[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        stats.append({
            'idx': idx,
            'area_um2': area_um2,
            'perim_um': perim_um,
            'feret_um': feret_um,
            'diam_eq_um': diam_eq_um,
        })
        idx += 1

    cv2.putText(img_color, f"Esporas: {len(stats)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return stats, img_color

def combinar_imagenes(original, roi_binario, roi_anotado, barra_y):
    # Marcar ROI en la imagen original
    img_roi_marcado = original.copy()
    cv2.rectangle(img_roi_marcado, (0, 0), (original.shape[1]-1, barra_y), (255, 255, 0), 2)

    alto_total = original.shape[0]
    ancho_total = original.shape[1]

    def preparar_lienzo(imagen_roi, alto_total, ancho_total):
        h_roi, w_roi = imagen_roi.shape[:2]
        scale = ancho_total / w_roi
        new_h = int(h_roi * scale)
        if new_h > alto_total:
            scale = alto_total / h_roi
            new_h = alto_total
            new_w = int(w_roi * scale)
        else:
            new_w = ancho_total
        img_res = cv2.resize(imagen_roi, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        if len(imagen_roi.shape) == 3:
            lienzo = np.zeros((alto_total, ancho_total, 3), dtype=np.uint8)
        else:
            lienzo = np.zeros((alto_total, ancho_total), dtype=np.uint8)
        lienzo[0:new_h, 0:new_w] = img_res
        return lienzo

    binario_lienzo = preparar_lienzo(roi_binario, alto_total, ancho_total)
    anotado_lienzo = preparar_lienzo(roi_anotado, alto_total, ancho_total)

    if len(binario_lienzo.shape) == 2:
        binario_lienzo = cv2.cvtColor(binario_lienzo, cv2.COLOR_GRAY2BGR)

    combinada = cv2.hconcat([img_roi_marcado, binario_lienzo, anotado_lienzo])
    return combinada

# ------------------ Funciones de análisis estadístico y gráficos ------------------
def analizar_y_graficar(datos_por_imagen, ruta_salida):
    """
    datos_por_imagen: dict {nombre_archivo: [lista de diámetros en µm]}
    """
    if not datos_por_imagen:
        print("No hay datos para análisis.")
        return

    # Preparar DataFrame para estadísticas y gráficos
    df_list = []
    for img, diametros in datos_por_imagen.items():
        for d in diametros:
            df_list.append({'Imagen': img, 'Diametro_um': d})
    df = pd.DataFrame(df_list)
    if df.empty:
        print("No hay datos de diámetros.")
        return

    # 1. Estadísticas descriptivas por imagen
    stats_df = df.groupby('Imagen')['Diametro_um'].agg(['count', 'mean', 'median', 'std', 'min', 'max']).reset_index()
    stats_df.columns = ['archivo', 'num_esporas', 'promedio_um', 'mediana_um', 'desv_um', 'min_um', 'max_um']
    # Guardar en CSV
    stats_path = os.path.join(ruta_salida, "resumen_estadisticas_por_imagen.csv")
    stats_df.to_csv(stats_path, index=False, encoding='utf-8')
    print(f"Estadísticas descriptivas guardadas en {stats_path}")

    # 2. Prueba de Kruskal-Wallis
    grupos = [group['Diametro_um'].values for name, group in df.groupby('Imagen')]
    if len(grupos) < 2:
        print("Se necesita al menos dos imágenes para comparación estadística.")
    else:
        h_stat, p_value = stats.kruskal(*grupos)
        print(f"\n=== Prueba de Kruskal-Wallis ===")
        print(f"H = {h_stat:.4f}, p = {p_value:.6f}")
        if p_value < 0.05:
            print("Diferencias significativas entre las imágenes (p < 0.05).")
            # Post-hoc de Dunn con corrección de Bonferroni
            posthoc = posthoc_dunn(df, val_col='Diametro_um', group_col='Imagen', p_adjust='bonferroni')
            posthoc_path = os.path.join(ruta_salida, "posthoc_dunn_bonferroni.csv")
            posthoc.to_csv(posthoc_path)
            print(f"Resultados del post-hoc de Dunn guardados en {posthoc_path}")
        else:
            print("No se encontraron diferencias significativas entre las imágenes.")

    # 3. Gráficos
    sns.set_style("whitegrid")
    # Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Imagen', y='Diametro_um', data=df, palette="Set3")
    plt.title("Distribución del diámetro equivalente circular por imagen")
    plt.ylabel("Diámetro (µm)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    boxplot_path = os.path.join(ruta_salida, "boxplot_diametros.png")
    plt.savefig(boxplot_path, dpi=300)
    plt.close()
    print(f"Boxplot guardado en {boxplot_path}")

    # Histogramas superpuestos
    plt.figure(figsize=(10, 6))
    for img in datos_por_imagen.keys():
        diametros = datos_por_imagen[img]
        sns.histplot(diametros, kde=True, label=img, alpha=0.5, bins=15)
    plt.title("Histograma de diámetros por imagen")
    plt.xlabel("Diámetro (µm)")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    hist_path = os.path.join(ruta_salida, "histogramas_diametros.png")
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"Histogramas guardados en {hist_path}")

    # Opcional: Violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Imagen', y='Diametro_um', data=df, palette="Set3")
    plt.title("Violin plot de diámetros por imagen")
    plt.ylabel("Diámetro (µm)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    violin_path = os.path.join(ruta_salida, "violinplot_diametros.png")
    plt.savefig(violin_path, dpi=300)
    plt.close()
    print(f"Violin plot guardado en {violin_path}")

# ------------------ Procesamiento principal ------------------
def procesar_carpeta(carpeta, longitud_real_um, subcarpeta_salida="procesadas"):
    ruta_salida = os.path.join(carpeta, subcarpeta_salida)
    os.makedirs(ruta_salida, exist_ok=True)

    datos_esporas = []          # lista de diccionarios (cada espora)
    diametros_por_imagen = {}   # para análisis estadístico

    resumen_por_imagen = []     # estadísticas básicas por imagen (se actualizará después)

    for archivo in os.listdir(carpeta):
        if not archivo.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
            continue
        ruta = os.path.join(carpeta, archivo)
        img_gray = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        img_color = cv2.imread(ruta)
        if img_gray is None or img_color is None:
            print(f"No se pudo leer {archivo}")
            continue

        pix_um, barra, img_color_con_barra = detectar_barra_escala(img_gray, img_color, longitud_real_um)
        if pix_um is None:
            print(f"No se detectó barra en {archivo}")
            continue
        x, y_barra, w, h = barra

        # Recortar ROI (por encima de la barra)
        roi_gray = img_gray[:y_barra, :]
        roi_color = img_color_con_barra[:y_barra, :].copy()
        if roi_gray.size == 0:
            print(f"ROI vacío en {archivo}")
            continue

        binario = binarizar_esporas(roi_gray)
        stats, img_anotada_roi = identificar_esporas(binario, roi_gray, pix_um, roi_color)

        # Guardar imágenes combinadas
        combinada = combinar_imagenes(img_color_con_barra, binario, img_anotada_roi, y_barra)
        nombre_base = os.path.splitext(archivo)[0]
        cv2.imwrite(os.path.join(ruta_salida, f"{nombre_base}_combinado.png"), combinada)
        cv2.imwrite(os.path.join(ruta_salida, f"{nombre_base}_esporas.png"), img_anotada_roi)

        # Recolectar datos
        diametros = []
        for s in stats:
            datos_esporas.append({
                'archivo': archivo,
                'id_espora': s['idx'],
                'area_um2': s['area_um2'],
                'perimetro_um': s['perim_um'],
                'feret_um': s['feret_um'],
                'diam_eq_um': s['diam_eq_um']
            })
            diametros.append(s['diam_eq_um'])

        diametros_por_imagen[archivo] = diametros

        n = len(stats)
        if n > 0:
            promedio = np.mean(diametros)
            mediana = np.median(diametros)
            desv = np.std(diametros)
            minimo = np.min(diametros)
            maximo = np.max(diametros)
        else:
            promedio = mediana = desv = minimo = maximo = 0

        resumen_por_imagen.append({
            'archivo': archivo,
            'pix_um': pix_um,
            'num_esporas': n,
            'promedio_um': promedio,
            'mediana_um': mediana,
            'desv_um': desv,
            'min_um': minimo,
            'max_um': maximo
        })
        print(f"Procesado {archivo} -> {pix_um:.4f} px/µm, {n} esporas")

    # Guardar CSV detallado por espora
    csv_detalle_path = os.path.join(ruta_salida, "resultados_esporas.csv")
    if datos_esporas:
        fieldnames = ['archivo', 'id_espora', 'area_um2', 'perimetro_um', 'feret_um', 'diam_eq_um']
        with open(csv_detalle_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(datos_esporas)
    else:
        with open(csv_detalle_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['archivo', 'id_espora', 'area_um2', 'perimetro_um', 'feret_um', 'diam_eq_um'])

    # Guardar resumen por imagen (estadísticas básicas)
    resumen_path = os.path.join(ruta_salida, "resumen_por_imagen.csv")
    with open(resumen_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['archivo', 'pix_um', 'num_esporas',
                                               'promedio_um', 'mediana_um', 'desv_um', 'min_um', 'max_um'])
        writer.writeheader()
        writer.writerows(resumen_por_imagen)

    # Análisis estadístico y gráficos (si hay datos)
    if diametros_por_imagen:
        analizar_y_graficar(diametros_por_imagen, ruta_salida)

    return datos_esporas, resumen_por_imagen, diametros_por_imagen

if __name__ == "__main__":
    carpeta = r"C:\Users\alain\OneDrive\Desktop\Myxomycetes\Lycogala\Muestra2\Escala 500" #Carpeta principal donde están las imágenes
    longitud_real_um = 50.0 #Longitud que aparece en la barra de escala en la imagen  SEM (en µm)
    datos, resumen, diam_por_img = procesar_carpeta(carpeta, longitud_real_um, subcarpeta_salida="procesadas")

    print("\n=== RESUMEN POR IMAGEN ===")
    for item in resumen:
        print(f"{item['archivo']}: {item['num_esporas']} esporas, "
              f"promedio = {item['promedio_um']:.2f} µm, mediana = {item['mediana_um']:.2f} µm, "
              f"desv = {item['desv_um']:.2f} µm, min = {item['min_um']:.2f}, max = {item['max_um']:.2f}")