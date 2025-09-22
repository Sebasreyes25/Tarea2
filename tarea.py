import argparse, math, time, random
from secrets import SystemRandom
import numpy as np
import matplotlib.pyplot as plt

def chi_square_uniform(u, bins=10):
    u = np.asarray(u)
    counts, _ = np.histogram(u, bins=bins, range=(0.0, 1.0))
    N = len(u)
    expected = N / bins
    chi2 = np.sum((counts - expected)**2 / expected)
    df = bins - 1
    from math import lgamma
    def _gammainc_lower(s, x, terms=200):
        total = 0.0
        term = 1.0 / s
        for n in range(terms):
            if n > 0:
                term *= x / (s + n)
            total += term
        return (x**s) * math.exp(-x) * total
    s = df/2.0
    x = chi2/2.0
    gamma_s = math.exp(lgamma(s))
    P = _gammainc_lower(s, x) / gamma_s 
    p_value = 1.0 - P
    return chi2, df, p_value, counts

def runs_test_above_below_median(u):
    u = np.asarray(u)
    med = np.median(u)
    seq = (u >= med).astype(int)
    runs = 1
    for i in range(1, len(seq)):
        if seq[i] != seq[i-1]:
            runs += 1
    n1 = int((seq == 1).sum())
    n2 = int((seq == 0).sum())
    if n1 == 0 or n2 == 0:
        return float('nan'), float('nan'), runs, n1, n2
    mean_runs = 1 + (2*n1*n2)/(n1+n2)
    var_runs = (2*n1*n2*(2*n1*n2 - n1 - n2)) / (((n1+n2)**2)*(n1+n2-1))
    z = (runs - mean_runs) / math.sqrt(var_runs) if var_runs > 0 else float('nan')
    p = 2 * (1 - 0.5*(1+math.erf(abs(z)/math.sqrt(2))))
    return z, p, runs, n1, n2

def gen_lcg(N, seed=12345, a=1664525, c=1013904223, m=2**32):
    X = seed % m
    out = []
    for _ in range(N):
        X = (a*X + c) % m
        out.append(X / m)
    return out

def gen_randu(N, seed=12345, a=65539, m=2**31):
    X = seed % m
    out = []
    for _ in range(N):
        X = (a*X) % m
        out.append(X / m)
    return out

def gen_middle_square(N, seed=675248, digits=6):
    out = []
    X = int(str(seed).zfill(digits)[-digits:])
    base = 10**digits
    for _ in range(N):
        Y = X*X
        Ystr = str(Y).zfill(2*digits)
        mid = Ystr[digits//2 : digits//2 + digits]
        X = int(mid)
        out.append(X / base)
    return out

def gen_mt(N, seed=42):
    rng = random.Random(seed)
    return [rng.random() for _ in range(N)]

def gen_bbs(N, seed=8731, p=10007, q=10039, bits_per_sample=24):
    assert p % 4 == 3 and q % 4 == 3, "p y q deben ser ≡ 3 (mod 4)"
    M = p*q
    x = (seed % M) or 3
    if math.gcd(x, M) != 1:
        x = (x + 1) % M
        if x == 0: x = 3
    out = []
    for _ in range(N):
        val = 0
        for _ in range(bits_per_sample):
            x = (x*x) % M
            bit = x & 1
            val = (val << 1) | bit
        out.append(val / (2**bits_per_sample))
    return out

def analyze_and_plot(u, title=""):
    t0 = time.time()
    chi2, df, pchi, _ = chi_square_uniform(u, bins=args.bins)
    z, pruns, runs, n1, n2 = runs_test_above_below_median(u)
    elapsed = (time.time() - t0)*1000

    print(f"\n {title} ")
    print(f"Muestras: {len(u)}, Bins: {args.bins}")
    print(f"Chi2: stat={chi2:.4f}, df={df}, p≈{pchi:.6f}")
    print(f"Rachas: z={z:.4f}, p≈{pruns:.6f}, runs={runs}, n1={n1}, n2={n2}")
    print(f"Tiempo análisis: {elapsed:.2f}")

    plt.figure()
    plt.hist(u, bins=args.bins, range=(0,1))
    plt.title(f"Histograma - {title}")
    plt.xlabel("u en [0,1)")
    plt.ylabel("Frecuencia")
    plt.tight_layout()

def mc_integral_uniform(f, a, b, N, rng_uniform_01):
    """∫_a^b f(x) dx ≈ (b-a) * (1/N) Σ f(a + (b-a)U), U~Unif(0,1)."""
    width = b - a
    n = 0
    mean_f = 0.0
    M2 = 0.0
    for _ in range(N):
        u = rng_uniform_01()
        x = a + width * u
        fx = f(x)
        n += 1
        delta = fx - mean_f
        mean_f += delta / n
        M2 += delta * (fx - mean_f)
    var_f = M2 / (n - 1) if n > 1 else 0.0
    I_hat = width * mean_f
    SE = width * math.sqrt(var_f / N)
    lo = I_hat - 1.96 * SE
    hi = I_hat + 1.96 * SE
    return I_hat, SE, (lo, hi), mean_f, var_f

def make_rng_from_list(lst):
    it = iter(lst)
    def _rng():
        try:
            return next(it)
        except StopIteration:
            return random.random()
    return _rng

def make_rng_algo(algo, N, seed, ms_digits=6, bbs_p=10007, bbs_q=10039, bbs_bits=24):
    if algo == "lcg":
        lst = gen_lcg(N, seed=seed)
    elif algo == "randu":
        lst = gen_randu(N, seed=seed)
    elif algo == "ms":
        lst = gen_middle_square(N, seed=seed, digits=ms_digits)
    elif algo == "mt":
        lst = gen_mt(N, seed=seed)
    elif algo == "bbs":
        lst = gen_bbs(N, seed=seed, p=bbs_p, q=bbs_q, bits_per_sample=bbs_bits)
    elif algo == "random":
        rng = random.Random(seed)
        lst = [rng.random() for _ in range(N)]
    else:
        raise ValueError("Algo de RNG no reconocido para MC.")
    return make_rng_from_list(lst)

def f_sin_pi(x):
    return math.sin(math.pi * x)

def f_std_norm_pdf(x):
    return math.exp(-x*x/2.0) / math.sqrt(2.0*math.pi)

def main(args):
    if args.mc:
        if args.f == "sin":
            f = f_sin_pi
            a_default, b_default = 0.0, 1.0
            real = 2.0 / math.pi
        else: 
            f = f_std_norm_pdf
            a_default, b_default = 0.0, 2.0
            real = 0.9772498680518208 - 0.5  

        a = a_default if args.a is None else args.a
        b = b_default if args.b is None else args.b

        rng = make_rng_algo(args.algo_mc, args.N, args.seed,
                            ms_digits=args.ms_digits,
                            bbs_p=args.bbs_p, bbs_q=args.bbs_q, bbs_bits=args.bbs_bits)

        I_hat, SE, (lo, hi), _, _ = mc_integral_uniform(f, a, b, args.N, rng)

        print(f"\n Monte Carlo: f={args.f}, intervalo [{a},{b}] ")
        print(f"N={args.N}, RNG={args.algo_mc}, seed={args.seed}")
        print(f"Estimación:  {I_hat:.9f}")
        print(f"SE (≈):      {SE:.9f}")
        print(f"IC 95%:      [{lo:.9f}, {hi:.9f}]")
        print(f"Valor real:  {real:.9f}")
        print(f"Error abs.:  {abs(I_hat-real):.9f}")
        return

    if args.algo == "lcg":
        u = gen_lcg(args.N, seed=args.seed)
        analyze_and_plot(u, "LCG")
    elif args.algo == "randu":
        u = gen_randu(args.N, seed=args.seed)
        analyze_and_plot(u, "RANDU")
    elif args.algo == "ms":
        digits = args.ms_digits
        u = gen_middle_square(args.N, seed=args.seed, digits=digits)
        analyze_and_plot(u, f"Middle-Square (d={digits})")
    elif args.algo == "mt":
        u = gen_mt(args.N, seed=args.seed)
        analyze_and_plot(u, "Mersenne Twister")
    elif args.algo == "bbs":
        u = gen_bbs(args.N, seed=args.seed, p=args.bbs_p, q=args.bbs_q, bits_per_sample=args.bbs_bits)
        analyze_and_plot(u, f"BBS (p={args.bbs_p}, q={args.bbs_q}, bits={args.bbs_bits})")
    else:
        raise ValueError("Algoritmo desconocido")

    if args.compare:
        u_mt = gen_mt(args.N, seed=args.seed+1)
        analyze_and_plot(u_mt, "random.Random")
        sysrng = SystemRandom()
        u_sec = [sysrng.random() for _ in range(args.N)]
        analyze_and_plot(u_sec, "secrets.SystemRandom ")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estudio de PRNGs + Monte Carlo")
    parser.add_argument("--algo", choices=["lcg","ms","mt","bbs","randu"], required=False,
                        help="Generador a usar (Parte 1)")
    parser.add_argument("-N", type=int, default=50000, help="Cantidad de muestras")
    parser.add_argument("--bins", type=int, default=10, help="Bins para histogramas/chi2")
    parser.add_argument("--seed", type=int, default=12345, help="Semilla")
    parser.add_argument("--compare", action="store_true",
                        help="Comparar contra random.Random y secrets.SystemRandom")
    parser.add_argument("--ms-digits", type=int, default=6,
                        help="Número de dígitos para middle-square")
    parser.add_argument("--bbs-p", type=int, default=10007, help="Primo p ≡ 3 (mod 4)")
    parser.add_argument("--bbs-q", type=int, default=10039, help="Primo q ≡ 3 (mod 4)")
    parser.add_argument("--bbs-bits", type=int, default=24,
                        help="Bits por muestra para BBS")
    parser.add_argument("--mc", action="store_true",
                        help="Ejecutar integración Monte Carlo")
    parser.add_argument("--f", choices=["sin","gauss"], default="sin",
                        help="Función a integrar: sin -> ∫_0^1 sin(pi x) dx; gauss -> ∫_0^2 N(0,1) dx")
    parser.add_argument("--a", type=float, default=None, help="Límite inferior (opcional)")
    parser.add_argument("--b", type=float, default=None, help="Límite superior (opcional)")
    parser.add_argument("--algo-mc", choices=["random","lcg","randu","ms","mt","bbs"], default="random",
                        help="RNG a usar en Monte Carlo")

    args = parser.parse_args()
    main(args)
