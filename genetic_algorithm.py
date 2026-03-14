import numpy as np

berat_item = np.array([2, 3, 4, 5, 9, 7, 6, 1], dtype=int)
nilai_item = np.array([3, 4, 8, 8, 10, 7, 6, 2], dtype=int)
kapasitas_tas = 15


ukuran_populasi = 30
jumlah_generasi = 80
probabilitas_crossover = 0.8
probabilitas_mutasi = 0.05
ukuran_turnamen = 3


def inisialisasi_populasi(generator_acak: np.random.Generator) -> np.ndarray:
    return generator_acak.integers(
        low=0,
        high=2,
        size=(ukuran_populasi, len(berat_item)),
        dtype=np.int8,
    )


def hitung_fitness(populasi: np.ndarray) -> np.ndarray:
    total_berat = populasi @ berat_item
    total_nilai = populasi @ nilai_item

    valid = total_berat <= kapasitas_tas
    skor_fitness = np.where(valid, total_nilai, 0)
    return skor_fitness


def seleksi_turnamen(
    populasi: np.ndarray,
    skor_fitness: np.ndarray,
    generator_acak: np.random.Generator,
) -> np.ndarray:
    indeks_kandidat = generator_acak.integers(
        low=0,
        high=ukuran_populasi,
        size=ukuran_turnamen,
    )

    indeks_kandidat_terbaik = indeks_kandidat[np.argmax(skor_fitness[indeks_kandidat])]
    return populasi[indeks_kandidat_terbaik].copy()


def crossover_satu_titik(
    induk_a: np.ndarray,
    induk_b: np.ndarray,
    generator_acak: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if generator_acak.random() >= probabilitas_crossover:
        return induk_a.copy(), induk_b.copy()

    titik_potong = generator_acak.integers(1, len(berat_item))

    anak_a = np.concatenate([induk_a[:titik_potong], induk_b[titik_potong:]])
    anak_b = np.concatenate([induk_b[:titik_potong], induk_a[titik_potong:]])
    return anak_a, anak_b


def mutasi(
    kromosom: np.ndarray,
    generator_acak: np.random.Generator,
) -> np.ndarray:
    mask_mutasi = generator_acak.random(len(kromosom)) < probabilitas_mutasi
    kromosom[mask_mutasi] = 1 - kromosom[mask_mutasi]
    return kromosom


def genetic_algorithm() -> None:
    generator_acak = np.random.default_rng()
    populasi_saat_ini = inisialisasi_populasi(generator_acak)

    solusi_terbaik = None
    fitness_terbaik = -1

    for _ in range(jumlah_generasi):
        skor_fitness = hitung_fitness(populasi_saat_ini)

        indeks_terbaik_generasi = int(np.argmax(skor_fitness))
        fitness_terbaik_generasi = int(skor_fitness[indeks_terbaik_generasi])
        if fitness_terbaik_generasi > fitness_terbaik:
            fitness_terbaik = fitness_terbaik_generasi
            solusi_terbaik = populasi_saat_ini[indeks_terbaik_generasi].copy()

        populasi_berikutnya = []

        while len(populasi_berikutnya) < ukuran_populasi:
            induk_a = seleksi_turnamen(populasi_saat_ini, skor_fitness, generator_acak)
            induk_b = seleksi_turnamen(populasi_saat_ini, skor_fitness, generator_acak)

            anak_a, anak_b = crossover_satu_titik(induk_a, induk_b, generator_acak)

            anak_a = mutasi(anak_a, generator_acak)
            anak_b = mutasi(anak_b, generator_acak)

            populasi_berikutnya.append(anak_a)
            if len(populasi_berikutnya) < ukuran_populasi:
                populasi_berikutnya.append(anak_b)

        populasi_saat_ini = np.array(populasi_berikutnya, dtype=np.int8)

    if solusi_terbaik is None:
        print("Tidak ada solusi yang ditemukan.")
        return

    total_berat_terbaik = int(solusi_terbaik @ berat_item)
    total_nilai_terbaik = int(solusi_terbaik @ nilai_item)
    indeks_item_terpilih = np.where(solusi_terbaik == 1)[0]

    print("Hasil Genetic Algorithm untuk Knapsack Problem")
    print(f"Kromosom terbaik      : {solusi_terbaik.tolist()}")
    print(f"Item terpilih (indeks): {indeks_item_terpilih.tolist()}")
    print(f"Total berat           : {total_berat_terbaik}")
    print(f"Total nilai           : {total_nilai_terbaik}")


genetic_algorithm()
