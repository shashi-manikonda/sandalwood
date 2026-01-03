import json
import os


def generate_cpp_header():
    """
    Generates a C++ header file with precomputed coefficients from JSON files.
    """
    json_dir = "src/mtflib/precomputed_coefficients_data"
    output_file = "src/mtflib/backends/cpp/precomputed_coefficients.hpp"

    coeff_map = {}
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            name = os.path.splitext(filename)[0].replace("_coefficients", "")
            with open(os.path.join(json_dir, filename), "r") as f:
                coeffs = json.load(f)
                coeff_map[name] = coeffs

    with open(output_file, "w") as f:
        f.write("#ifndef PRECOMPUTED_COEFFICIENTS_HPP\n")
        f.write("#define PRECOMPUTED_COEFFICIENTS_HPP\n\n")
        f.write("#include <map>\n")
        f.write("#include <string>\n")
        f.write("#include <vector>\n\n")
        f.write("namespace mtf_coeffs {\n\n")
        f.write(
            "static std::map<std::string, "
            "std::vector<double>> precomputed_coefficients = {\n"
        )

        for i, (name, coeffs) in enumerate(coeff_map.items()):
            f.write(f'    {{"{name}", {{')
            f.write(", ".join(map(str, coeffs)))
            f.write("}}")
            if i < len(coeff_map) - 1:
                f.write(",\n")
            else:
                f.write("\n")

        f.write("};\n\n")
        f.write("} // namespace mtf_coeffs\n\n")
        f.write("#endif // PRECOMPUTED_COEFFICIENTS_HPP\n")

    print(f"Generated {output_file}")


if __name__ == "__main__":
    generate_cpp_header()
