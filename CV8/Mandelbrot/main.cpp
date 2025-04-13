#include <iostream>
#include <tuple>
#include <sstream>
#include <chrono>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "BS_thread_pool.hpp"
#include "qd_single.h" // Quad double library for higher precision and therefore bigger zoom

// Initial view parameters
const double REAL_MIN = -2.0;
const double REAL_MAX = +1.0;
const double IMAG_MIN = -1.5;
const double IMAG_MAX = +1.5;

// Image size
const int WIDTH = 800;
const int HEIGHT = 800;

// Color conversion from HSV to RGB to make the image look nice :)
std::tuple<unsigned char, unsigned char, unsigned char> hsv_to_rgb(float h, float s, float v) {
    float c = v * s;
    float x = c * (1 - fabs(fmod(h / 60.0, 2) - 1));
    float m = v - c;
    float r, g, b;

    if (h < 60) { r = c; g = x; b = 0; }
    else if (h < 120) { r = x; g = c; b = 0; }
    else if (h < 180) { r = 0; g = c; b = x; }
    else if (h < 240) { r = 0; g = x; b = c; }
    else if (h < 300) { r = x; g = 0; b = c; }
    else { r = c; g = 0; b = x; }

    return {
        (unsigned char)((r + m) * 255),
        (unsigned char)((g + m) * 255),
        (unsigned char)((b + m) * 255)
    };
}

void save_image(const char* filename, float* mandelbrot_set) {
	unsigned char* image_data = new unsigned char[WIDTH * HEIGHT * 3];

    // Create a color map
	for (int i = 0; i < HEIGHT; ++i) {
		for (int j = 0; j < WIDTH; ++j) {
			float t = mandelbrot_set[i * WIDTH + j];

            if (t == 0) {
                image_data[(i * WIDTH + j) * 3 + 0] = 0;
                image_data[(i * WIDTH + j) * 3 + 1] = 0;
                image_data[(i * WIDTH + j) * 3 + 2] = 0;
                continue;
            }

            // Use t as hue value for color mapping
            float hue = fmod(360.0 * t, 360.0);
            float sat = 1.0f;
            float val = 1.0f;

			auto [r, g, b] = hsv_to_rgb(hue, sat, val);

			// Set pixel color
			image_data[(i * WIDTH + j) * 3 + 0] = r;
			image_data[(i * WIDTH + j) * 3 + 1] = g;
			image_data[(i * WIDTH + j) * 3 + 2] = b;
		}
	}

	// Save the image using stb_image_write
	stbi_write_png(filename, WIDTH, HEIGHT, 3, image_data, WIDTH * 3);
    delete[] image_data;
}


// Function to compute the Mandelbrot set
void compute_mandelbrot(qd_real real_min, qd_real real_max, qd_real imag_min, qd_real imag_max, qd_real zoom, int frame) {
	auto start = std::chrono::high_resolution_clock::now();

    // Array of iterations gives only 256 distinct colors
    // Use time smoothing to get smoother gradients
	float* mandelbrot_set = new float[HEIGHT * WIDTH];

    qd_real delta_x = (real_max - real_min) / (WIDTH - 1);
    qd_real delta_y = (imag_max - imag_min) / (HEIGHT - 1);

    // Dynamically adjust iterations based on zoom level
    int max_iterations = 100 + to_int(log(zoom) / log(qd_real(2.0))) * 100;

    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j) {

            // Map current pixel coordinates to complex number c = x + yi
            qd_real c_real = real_min + j * delta_x;
            qd_real c_imag = imag_max - i * delta_y;

            qd_real z_real = 0.0, z_imag = 0.0; // Starting point for the iteration
            int iter_count = 0;

            // Iterate until divergence or maximum iterations reached
            while (iter_count < max_iterations) {

                // z_new = z_prev^2 + c
                qd_real new_z_real = z_real * z_real - z_imag * z_imag + c_real;
                qd_real new_z_imag = 2.0 * z_real * z_imag + c_imag;

                // Check if the point has diverged
				qd_real mag2 = new_z_real * new_z_real + new_z_imag * new_z_imag;
                if (mag2 > 4.0) {
                    // Escape
                    double log_zn = to_double(log(mag2) / 2.0);
                    double nu = log(log_zn / log(2.0)) / log(2.0);
                    mandelbrot_set[i * WIDTH + j] = (iter_count + 1 - nu) / max_iterations; // Smoothing
                    break;
                }

                // Update z for next iteration and increment counter
                z_real = new_z_real;
                z_imag = new_z_imag;
                ++iter_count;
            }

			// If the point did not escape, color will be black
			if (iter_count == max_iterations) {
				mandelbrot_set[i * WIDTH + j] = 0;
			}
        }
    }

    std::ostringstream oss;
    oss << std::setw(5) << std::setfill('0') << frame;
    std::string filename = "render/_" + oss.str() + ".png";

	save_image(filename.c_str(), mandelbrot_set);
    
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Frame " << frame << " computed in " << elapsed.count() << " seconds" << std::endl;
}

int main() {
    // Create thread pool
	BS::light_thread_pool pool(32);

    // Parameters for zooming
    qd_real zoom_point_x = -0.743643887037158704752191506114774;
    qd_real zoom_point_y = 0.131825904205311970493132056385139;
	qd_real zoom_per_frame = 1.05;


    double real_width = REAL_MAX - REAL_MIN;
    double imag_height = IMAG_MAX - IMAG_MIN;

    // Each frame of zoom is handled by a new thread
    for (int i = 32 * 20; i < 32 * 24; ++i) {
        qd_real current_zoom = pow(zoom_per_frame, i);

        // New dimensions after scaling
		qd_real current_width = real_width / current_zoom;
		qd_real current_height = imag_height / current_zoom;

        // Update boundaries centered at (zoom_point_x, zoom_point_y)
        qd_real real_min = zoom_point_x - current_width / 2.0;
        qd_real real_max = zoom_point_x + current_width / 2.0;
        qd_real imag_min = zoom_point_y - current_height / 2.0;
        qd_real imag_max = zoom_point_y + current_height / 2.0;

		pool.submit_task([=]() {
			compute_mandelbrot(real_min, real_max, imag_min, imag_max, current_zoom, i);
			});
    }

    return 0;
}