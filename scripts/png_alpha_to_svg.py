#!/usr/bin/env python3
import struct
import sys
import zlib
from collections import defaultdict
from pathlib import Path


PNG_SIG = b"\x89PNG\r\n\x1a\n"


def paeth(a, b, c):
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def read_png_rgba(path):
    data = Path(path).read_bytes()
    if not data.startswith(PNG_SIG):
        raise ValueError("not a PNG file")

    pos = len(PNG_SIG)
    width = height = bit_depth = color_type = None
    idat = bytearray()

    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        chunk_type = data[pos + 4 : pos + 8]
        chunk_data = data[pos + 8 : pos + 8 + length]
        pos += 12 + length

        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", chunk_data)
        elif chunk_type == b"IDAT":
            idat.extend(chunk_data)
        elif chunk_type == b"IEND":
            break

    if bit_depth != 8 or color_type != 6:
        raise ValueError(f"expected 8-bit RGBA PNG, got bit_depth={bit_depth}, color_type={color_type}")

    raw = zlib.decompress(bytes(idat))
    bpp = 4
    stride = width * bpp
    rows = []
    offset = 0
    prev = [0] * stride

    for _ in range(height):
        filter_type = raw[offset]
        offset += 1
        scan = list(raw[offset : offset + stride])
        offset += stride
        recon = [0] * stride

        for i, val in enumerate(scan):
            left = recon[i - bpp] if i >= bpp else 0
            up = prev[i]
            up_left = prev[i - bpp] if i >= bpp else 0

            if filter_type == 0:
                recon[i] = val
            elif filter_type == 1:
                recon[i] = (val + left) & 255
            elif filter_type == 2:
                recon[i] = (val + up) & 255
            elif filter_type == 3:
                recon[i] = (val + ((left + up) // 2)) & 255
            elif filter_type == 4:
                recon[i] = (val + paeth(left, up, up_left)) & 255
            else:
                raise ValueError(f"unsupported PNG filter {filter_type}")

        rows.append(recon)
        prev = recon

    return width, height, rows


def rdp(points, epsilon):
    if len(points) < 4:
        return points

    def perpendicular_distance(point, start, end):
        x, y = point
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
        return abs(dy * x - dx * y + x2 * y1 - y2 * x1) / ((dx * dx + dy * dy) ** 0.5)

    closed = points[0] == points[-1]
    work = points[:-1] if closed else points

    def simplify(seq):
        if len(seq) < 3:
            return seq
        start, end = seq[0], seq[-1]
        max_dist = -1
        index = 0
        for i in range(1, len(seq) - 1):
            dist = perpendicular_distance(seq[i], start, end)
            if dist > max_dist:
                index = i
                max_dist = dist
        if max_dist > epsilon:
            return simplify(seq[: index + 1])[:-1] + simplify(seq[index:])
        return [start, end]

    if closed:
        # Start at a corner far from the centroid so a closed curve does not simplify against a zero-length segment.
        cx = sum(x for x, _ in work) / len(work)
        cy = sum(y for _, y in work) / len(work)
        split = max(range(len(work)), key=lambda i: (work[i][0] - cx) ** 2 + (work[i][1] - cy) ** 2)
        rotated = work[split:] + work[:split] + [work[split]]
        out = simplify(rotated)
        return out if out[0] == out[-1] else out + [out[0]]

    return simplify(work)


def remove_collinear(points):
    if len(points) < 4:
        return points
    out = []
    closed = points[0] == points[-1]
    work = points[:-1] if closed else points
    n = len(work)
    for i, point in enumerate(work):
        prev = work[(i - 1) % n]
        nxt = work[(i + 1) % n]
        if (point[0] - prev[0]) * (nxt[1] - point[1]) == (point[1] - prev[1]) * (nxt[0] - point[0]):
            continue
        out.append(point)
    if closed and out:
        out.append(out[0])
    return out


def mask_to_loops(mask, width, height):
    edges = defaultdict(list)

    def filled(x, y):
        return 0 <= x < width and 0 <= y < height and mask[y][x]

    for y in range(height):
        for x in range(width):
            if not mask[y][x]:
                continue
            if not filled(x, y - 1):
                edges[(x, y)].append((x + 1, y))
            if not filled(x + 1, y):
                edges[(x + 1, y)].append((x + 1, y + 1))
            if not filled(x, y + 1):
                edges[(x + 1, y + 1)].append((x, y + 1))
            if not filled(x - 1, y):
                edges[(x, y + 1)].append((x, y))

    loops = []
    while edges:
        start = next(iter(edges))
        current = start
        loop = [current]
        guard = 0

        while True:
            guard += 1
            if guard > width * height * 8:
                raise RuntimeError("loop tracing guard exceeded")
            next_points = edges.get(current)
            if not next_points:
                break
            nxt = next_points.pop()
            if not next_points:
                del edges[current]
            current = nxt
            loop.append(current)
            if current == start:
                break

        if len(loop) > 3 and loop[0] == loop[-1]:
            loops.append(loop)

    return loops


def loop_area(loop):
    area = 0
    for (x1, y1), (x2, y2) in zip(loop, loop[1:]):
        area += x1 * y2 - x2 * y1
    return area / 2


def path_from_loop(loop):
    parts = [f"M{loop[0][0]} {loop[0][1]}"]
    for x, y in loop[1:-1]:
        parts.append(f"L{x} {y}")
    parts.append("Z")
    return " ".join(parts)


def main():
    if len(sys.argv) < 3:
        print("usage: png_alpha_to_svg.py input.png output.svg [threshold=16] [epsilon=1.35]", file=sys.stderr)
        sys.exit(2)

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    epsilon = float(sys.argv[4]) if len(sys.argv) > 4 else 1.35

    width, height, rows = read_png_rgba(src)
    mask = []
    for row in rows:
        mask_row = []
        for x in range(width):
            alpha = row[x * 4 + 3]
            mask_row.append(alpha >= threshold)
        mask.append(mask_row)

    loops = mask_to_loops(mask, width, height)
    loops = sorted(loops, key=lambda item: abs(loop_area(item)), reverse=True)

    simplified = []
    for loop in loops:
        if abs(loop_area(loop)) < 12:
            continue
        loop = remove_collinear(loop)
        loop = rdp(loop, epsilon)
        simplified.append(loop)

    paths = "\n    ".join(f'<path d="{path_from_loop(loop)}" />' for loop in simplified)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="GeekOnUp logo">
  <g fill="currentColor" fill-rule="evenodd">
    {paths}
  </g>
</svg>
'''
    dst.write_text(svg)
    print(f"wrote {dst} ({len(simplified)} paths, {sum(len(loop) for loop in simplified)} points)")


if __name__ == "__main__":
    main()
