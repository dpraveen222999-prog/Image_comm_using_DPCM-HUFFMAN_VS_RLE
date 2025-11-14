import cv2
import numpy as np
from collections import Counter
import heapq
import math
import os
import struct

# ============================================================
# HUFFMAN CODING
# ============================================================
class Node:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None
    def __lt__(self, other):
        return self.freq < other.freq

def build_tree(freq):
    heap = [Node(k, v) for k, v in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        merged = Node(None, n1.freq + n2.freq)
        merged.left = n1
        merged.right = n2
        heapq.heappush(heap, merged)
    return heap[0]

def generate_codes(node, prefix="", mapping=None):
    if mapping is None:
        mapping = {}
    if node.symbol is not None:
        mapping[node.symbol] = prefix
    else:
        generate_codes(node.left, prefix + "0", mapping)
        generate_codes(node.right, prefix + "1", mapping)
    return mapping

def huffman_encode(data):
    freq = Counter(data)
    root = build_tree(freq)
    code_map = generate_codes(root)
    encoded = "".join(code_map[x] for x in data)
    return encoded, code_map


# ============================================================
# RLE CODING
# ============================================================
def rle_encode(data):
    encoded = []
    prev = data[0]
    cnt = 1
    for i in range(1, len(data)):
        if data[i] == prev:
            cnt += 1
        else:
            encoded.append((prev, cnt))
            prev = data[i]
            cnt = 1
    encoded.append((prev, cnt))
    return encoded


# ============================================================
# DPCM + QUANTIZATION
# ============================================================
def quantize(err, levels=32):
    e_min, e_max = -255, 255
    step = (e_max - e_min) / levels
    q = np.floor((err - e_min) / step)
    q = np.clip(q, 0, levels - 1)
    return e_min + q * step + step / 2

def dpcm_encode(img, levels=32):
    h, w = img.shape
    pred = np.zeros((h, w))
    qerr = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if i == 0 and j == 0:
                p = 0
            elif i == 0:
                p = pred[i, j - 1]
            elif j == 0:
                p = pred[i - 1, j]
            else:
                p = (pred[i - 1, j] + pred[i, j - 1]) / 2

            err = img[i, j] - p
            q = quantize(err, levels)
            qerr[i, j] = q
            pred[i, j] = p + q
    return qerr

def dpcm_decode(qerr):
    h, w = qerr.shape
    rec = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if i == 0 and j == 0:
                p = 0
            elif i == 0:
                p = rec[i, j - 1]
            elif j == 0:
                p = rec[i - 1, j]
            else:
                p = (rec[i - 1, j] + rec[i, j - 1]) / 2
            rec[i, j] = p + qerr[i, j]
    return rec


# ============================================================
# METRICS
# ============================================================
def mse(o, r):
    return np.mean((o - r)**2)

def psnr(o, r):
    m = mse(o, r)
    if m == 0:
        return float("inf")
    return 20 * math.log10(255 / math.sqrt(m))


# ============================================================
# MAIN
# ============================================================
path = input("Enter image path: ").strip()

if not os.path.exists(path):
    print("File does not exist!")
    exit()

img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

orig_bits = img.size * 8
print("\nOriginal Bits:", orig_bits)

# Convert channels to float
R = img[:,:,0].astype(np.float64)
G = img[:,:,1].astype(np.float64)
B = img[:,:,2].astype(np.float64)

# ===========================
# DPCM + HUFFMAN
# ===========================
print("\n--- DPCM + HUFFMAN ---")

qR = dpcm_encode(R)
qG = dpcm_encode(G)
qB = dpcm_encode(B)

flatR = qR.flatten().tolist()
flatG = qG.flatten().tolist()
flatB = qB.flatten().tolist()

encR, mapR = huffman_encode(flatR)
encG, mapG = huffman_encode(flatG)
encB, mapB = huffman_encode(flatB)

# Save actual compressed bitstream (.bin)
with open("compressed_dpcm_huffman.bin", "wb") as f:
    f.write(int(encR + encG + encB, 2).to_bytes((len(encR+encG+encB)+7)//8, byteorder='big'))

huff_bits = len(encR) + len(encG) + len(encB)

print("Bits After Huffman:", huff_bits)
print("Compression Ratio:", orig_bits / huff_bits)

# Decode and reconstruct
dR = np.array(flatR).reshape(R.shape)  
dG = np.array(flatG).reshape(G.shape)
dB = np.array(flatB).reshape(B.shape)

recR = dpcm_decode(dR)
recG = dpcm_decode(dG)
recB = dpcm_decode(dB)

rec_huff = np.dstack([recR, recG, recB]).clip(0,255).astype(np.uint8)
cv2.imwrite("reconstructed_dpcm.jpg",
            cv2.cvtColor(rec_huff, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 90])

MSE_h = mse(img, rec_huff)
PSNR_h = psnr(img, rec_huff)

# ===========================
# RLE
# ===========================
print("\n--- RLE ---")

flat = img.flatten().tolist()
rle = rle_encode(flat)

# Save RLE to binary
with open("compressed_rle.bin", "wb") as f:
    for val, cnt in rle:
        f.write(struct.pack('HI', val, cnt))

rle_bits = len(rle) * (8 + 16)

print("Bits After RLE:", rle_bits)
print("Compression Ratio:", orig_bits / rle_bits)

rec_rle = np.array(flat).reshape(img.shape).astype(np.uint8)
cv2.imwrite("reconstructed_rle.jpg",
            cv2.cvtColor(rec_rle, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 90])

MSE_r = mse(img, rec_rle)
PSNR_r = psnr(img, rec_rle)

# ===========================
# SUMMARY
# ===========================
print("\n--- SUMMARY ---")
print("Method\t\tBits\tCR\tPSNR\tMSE")
print(f"DPCM+Huff\t{huff_bits}\t{orig_bits/huff_bits:.2f}\t{PSNR_h:.2f}\t{MSE_h:.5f}")
print(f"RLE\t\t{rle_bits}{orig_bits/rle_bits:.2f}\t{PSNR_r:.2f}\t{MSE_r:.5f}")

print("\nSaved files:")
print(" compressed_dpcm_huffman.bin")
print(" compressed_rle.bin")
print(" reconstructed_dpcm.png")
print(" reconstructed_rle.png")
