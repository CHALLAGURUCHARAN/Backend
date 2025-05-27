from mobius_strip import MobiusStrip

def main():
    mobius = MobiusStrip(R=1.0, w=0.5, n=200)
    area = mobius.compute_surface_area()
    length = mobius.compute_edge_length()
    print(f"Surface Area: {area:.4f} units²")
    print(f"Edge Length: {length:.4f} units")
    mobius.plot()

if __name__ == "__main__":
    main()