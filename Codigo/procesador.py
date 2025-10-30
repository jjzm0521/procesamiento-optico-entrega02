import os
import zipfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ====================================================================
# CONSTANTES GLOBALES
# ====================================================================

LONGITUD_ONDA = 633e-9      
import os
import zipfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

LONGITUD_ONDA = 633e-9
DISTANCIA_FOCAL = 500e-3
DIAMETRO_L1 = 100e-3
DIAMETRO_M2 = 50e-3
ANCHO_M1 = 10.4e-3
ALTO_M1 = 5.8e-3
CAM1_PIXELES_X = 4640
CAM1_PIXELES_Y = 3506
TAMANO_PIXEL_CAM1 = 3.8e-6
CAM2_PIXELES_X = 1280
CAM2_PIXELES_Y = 1024
TAMANO_PIXEL_CAM2 = 5.2e-6

DIST_OBJ_BS = 250e-3
DIST_BS_ESPEJO = 100e-3
DIST_ESPEJO_LENTE = 250e-3

N = 1024
TAMANO_PLANO_ENTRADA = 20e-3

def propagar_abcd(campo_entrada: np.ndarray, dx_entrada: float, longitud_onda: float, A: float, B: float, C: float, D: float) -> tuple:
    """Propaga un campo óptico usando la matriz ABCD."""
    if not isinstance(campo_entrada, np.ndarray):
        raise TypeError("campo_entrada debe ser un array de NumPy.")
    if campo_entrada.shape[0] != campo_entrada.shape[1]:
        raise ValueError("El campo de entrada debe ser una matriz cuadrada.")
    if B == 0:
        return campo_entrada, dx_entrada

    N = campo_entrada.shape[0]
    k = 2 * np.pi / longitud_onda
    x = (np.arange(N) - N // 2) * dx_entrada
    X, Y = np.meshgrid(x, x)
    fase_inicial = np.exp(1j * k / (2 * B) * A * (X**2 + Y**2))
    campo_modificado = campo_entrada * fase_inicial
    campo_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(campo_modificado)))
    dx_salida = longitud_onda * abs(B) / (N * dx_entrada)
    x_salida = (np.arange(N) - N // 2) * dx_salida
    Xs, Ys = np.meshgrid(x_salida, x_salida)
    fase_final = np.exp(1j * k / (2 * B) * D * (Xs**2 + Ys**2))
    factor_escala = np.exp(1j * k * B) / (1j * longitud_onda * abs(B))
    campo_salida = factor_escala * campo_fft * fase_final * (dx_entrada ** 2)
    return campo_salida, dx_salida

def crear_filtro_muesca_adaptivo(campo_fourier, n_muescas_max=10, radio_muesca_pix=15, orden_butter=5, radio_exclusion_dc_pix=10, radio_min_pix=15, radio_max_pix=None, percentil_det=90, N=1024):
    F = campo_fourier
    N = F.shape[0]
    assert F.shape[0] == F.shape[1], "Se asume cuadrado."
    I = np.abs(F)**2
    L = np.log1p(I)
    win = np.hanning(N)
    W = np.outer(win, win)
    Lw = L * W
    thr = np.percentile(Lw, percentil_det)
    cand = np.argwhere(Lw > thr)
    cy = cx = N // 2
    yy, xx = np.mgrid[0:N, 0:N]
    rr = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    if radio_max_pix is None:
        radio_max_pix = N/2 - 4
    mask_rango = (rr > max(radio_min_pix, radio_exclusion_dc_pix)) & (rr < radio_max_pix)
    cand = [tuple(p) for p in cand if mask_rango[p[0], p[1]]]
    cand.sort(key=lambda p: Lw[p[0], p[1]], reverse=True)
    elegidos = []
    tomado = np.zeros((N, N), dtype=bool)
    supresion = max(6, radio_muesca_pix)
    for (y, x) in cand:
        if len(elegidos) >= n_muescas_max:
            break
        if tomado[y, x]:
            continue
        yc = (2*cy - y) % N
        xc = (2*cx - x) % N
        elegidos.append((y, x))
        elegidos.append((yc, xc))
        y0, y1 = max(0, y - supresion), min(N, y + supresion + 1)
        x0, x1 = max(0, x - supresion), min(N, x + supresion + 1)
        tomado[y0:y1, x0:x1] = True
        y0c, y1c = max(0, yc - supresion), min(N, yc + supresion + 1)
        x0c, x1c = max(0, xc - supresion), min(N, xc + supresion + 1)
        tomado[y0c:y1c, x0c:x1c] = True
    H = np.ones((N, N), dtype=float)
    def butter_notch(u, v, u0, v0, D0, n):
        Dk = np.sqrt((u - u0)**2 + (v - v0)**2)
        Dkc = np.sqrt((u + u0)**2 + (v + v0)**2)
        return 1.0 / (1.0 + (D0**2 / (Dk * Dkc + 1e-12))**n)
    U = xx - cx
    V = yy - cy
    for (y, x) in elegidos:
        u0 = x - cx
        v0 = y - cy
        H *= butter_notch(U, V, u0, v0, radio_muesca_pix, orden_butter)
    H[rr < radio_exclusion_dc_pix] = 1.0
    return H

def cargar_imagen_objeto(ruta_archivo, tamano_pixeles):
    try:
        img = Image.open(ruta_archivo).convert('L')
        img = img.resize((tamano_pixeles, tamano_pixeles), Image.Resampling.LANCZOS)
        img_array = np.array(img)
        if np.max(img_array) > 0:
            img_array = img_array / 255.0
        return img_array
    except Exception as e:
        print(f"Error cargando {ruta_archivo}: {e}")
        return None

def crear_apertura_rectangular(ancho, alto, tamano_m, N_pixeles):
    coordenadas = np.linspace(-tamano_m / 2, tamano_m / 2, N_pixeles)
    X, Y = np.meshgrid(coordenadas, coordenadas)
    apertura = np.zeros((N_pixeles, N_pixeles))
    apertura[(np.abs(X) < ancho / 2) & (np.abs(Y) < alto / 2)] = 1
    return apertura

def procesar_imagen_individual(ruta_imagen, nombre_salida, carpeta_resultados):
    try:
        print(f"Procesando: {nombre_salida}...", end=" ", flush=True)
        S = cargar_imagen_objeto(ruta_imagen, N)
        if S is None:
            return {'exito': False, 'error': 'No se pudo cargar la imagen'}
        dx = TAMANO_PLANO_ENTRADA / N
        coordenadas = np.linspace(-TAMANO_PLANO_ENTRADA / 2, TAMANO_PLANO_ENTRADA / 2, N)
        A1, B1, C1, D1 = 0, DISTANCIA_FOCAL, -1/DISTANCIA_FOCAL, 0
        campo_en_M1, dx_M1 = propagar_abcd(S, dx, LONGITUD_ONDA, A1, B1, C1, D1)
        tamano_plano_M1 = N * dx_M1
        mascara_fisica_espejo = crear_apertura_rectangular(ANCHO_M1, ALTO_M1, tamano_plano_M1, N)
        filtro_muesca = crear_filtro_muesca_adaptivo(campo_en_M1)
        mascara_espejo = mascara_fisica_espejo * filtro_muesca
        campo_despues_M1 = campo_en_M1 * mascara_espejo
        A2, B2, C2, D2 = 0, DISTANCIA_FOCAL, -1/DISTANCIA_FOCAL, 0
        O, dx_cam1 = propagar_abcd(campo_despues_M1, dx_M1, LONGITUD_ONDA, A2, B2, C2, D2)
        intensidad_cam1 = np.abs(O)**2
        if np.max(intensidad_cam1) > 0:
            intensidad_cam1 /= np.max(intensidad_cam1)
        d_entrada = DIST_OBJ_BS + DIST_BS_ESPEJO
        M_prop_entrada = np.array([[1, d_entrada], [0, 1]])
        M_espejo_plano = np.array([[1, 0], [0, 1]])
        M_prop_intermedia = np.array([[1, DIST_ESPEJO_LENTE], [0, 1]])
        M_lente = np.array([[1, 0], [-1/DISTANCIA_FOCAL, 1]])
        M_prop_salida = np.array([[1, DISTANCIA_FOCAL], [0, 1]])
        M_total_S_U = M_prop_salida @ M_lente @ M_prop_intermedia @ M_espejo_plano @ M_prop_entrada
        A3, B3, C3, D3 = M_total_S_U[0,0], M_total_S_U[0,1], M_total_S_U[1,0], M_total_S_U[1,1]
        campo_U, dx_cam2 = propagar_abcd(S, dx, LONGITUD_ONDA, A3, B3, C3, D3)
        intensidad_cam2 = np.abs(campo_U)**2
        if np.max(intensidad_cam2) > 0:
            intensidad_cam2 = np.log(1e-9 + intensidad_cam2)
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Simulación del Procesador Óptico - {nombre_salida}', fontsize=18)
        rango_entrada_mm = TAMANO_PLANO_ENTRADA / 2e-3
        axes[0, 0].imshow(S, cmap='gray', extent=[-rango_entrada_mm, rango_entrada_mm, -rango_entrada_mm, rango_entrada_mm])
        axes[0, 0].set_title('Objeto de Entrada S(ξ, η)')
        axes[0, 0].set_xlabel('ξ (mm)')
        axes[0, 0].set_ylabel('η (mm)')
        rango_m1_mm = tamano_plano_M1 / 2e-3
        axes[1, 0].imshow(mascara_espejo, cmap='gray', extent=[-rango_m1_mm, rango_m1_mm, -rango_m1_mm, rango_m1_mm])
        axes[1, 0].set_title('Máscara Mejorada t(x, y)')
        axes[1, 0].set_xlabel('x (mm)')
        axes[1, 0].set_ylabel('y (mm)')
        intensidad_espejo = np.abs(campo_en_M1)**2
        intensidad_espejo_log = np.log(1e-9 + intensidad_espejo)
        axes[0, 1].imshow(intensidad_espejo_log, cmap='hot', extent=[-rango_m1_mm, rango_m1_mm, -rango_m1_mm, rango_m1_mm])
        axes[0, 1].set_title('Intensidad en el Espejo M1 (antes de t)')
        axes[0, 1].set_xlabel('x (mm)')
        axes[0, 1].set_ylabel('y (mm)')
        rango_cam2_mm = (N * dx_cam2 * 1e3) / 2
        axes[1, 1].imshow(intensidad_cam2, cmap='hot', extent=[-rango_cam2_mm, rango_cam2_mm, -rango_cam2_mm, rango_cam2_mm])
        axes[1, 1].set_title("Resultado en Cámara 2: U(x', y')")
        axes[1, 1].set_xlabel("x' (mm)")
        axes[1, 1].set_ylabel("y' (mm)")
        rango_cam1_mm = (N * dx_cam1 * 1e3) / 2
        axes[0, 2].imshow(intensidad_cam1, cmap='gray', extent=[-rango_cam1_mm, rango_cam1_mm, -rango_cam1_mm, rango_cam1_mm])
        axes[0, 2].set_title('Resultado en Cámara 1: O(u, v)')
        axes[0, 2].set_xlabel('u (mm)')
        axes[0, 2].set_ylabel('v (mm)')
        axes[1, 2].axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        nombre_base = Path(nombre_salida).stem
        ruta_figura = os.path.join(carpeta_resultados, f"{nombre_base}_resultado.png")
        plt.savefig(ruta_figura, dpi=100, bbox_inches='tight')
    plt.close(fig)


def procesar_zip(ruta_zip):
    temp_dir = 'temp_imagenes'
    resultados_dir = 'resultados_procesamiento'
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(resultados_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(ruta_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        print("ZIP extraído correctamente\n")
    except Exception as e:
        print(f"Error extrayendo ZIP: {e}")
        return []
    extensiones = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.PNG', '*.JPG', '*.JPEG', '*.BMP', '*.TIFF')
    imagenes = []
    for ext in extensiones:
        for archivo in Path(temp_dir).rglob(ext):
            imagenes.append(archivo)
    imagenes = sorted(list(set(imagenes)))
    print(f"ENCONTRADAS {len(imagenes)} IMAGEN(ES)\n")
    resultados = []
    for idx, ruta_img in enumerate(imagenes, 1):
        print(f"[{idx}/{len(imagenes)}] {ruta_img.name:<50} ", end="", flush=True)
        resultado = procesar_imagen_individual(str(ruta_img), ruta_img.name, resultados_dir)
        resultados.append({'nombre': ruta_img.name, 'ruta': str(ruta_img), 'resultado': resultado})
    return resultados, resultados_dir, imagenes, temp_dir


def __main__():
    ruta_zip = r'datos_prueba\Noise images.zip'
    resultados, carpeta_resultados, imagenes, temp_dir = procesar_zip(ruta_zip)


if __name__ == '__main__':
    __main__()