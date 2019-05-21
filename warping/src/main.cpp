// main source file for data preprocessing and face alignment
#include <math.h>
#include <igl/readOBJ.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <igl/writeOBJ.h>
#include <igl/boundary_loop.h>
#include <igl/slice.h>
#include <igl/AABB.h>
#include <igl/per_face_normals.h>
#include <igl/cotmatrix.h>
#include <igl/cat.h>
#include <fstream>

using namespace Eigen;
using namespace std;
using Viewer = igl::opengl::glfw::Viewer;

// scanned face data
MatrixXd V_template;
MatrixXd N_template;
MatrixXi F_template;
VectorXi landmarks_template;
MatrixXd landmarks_template_points;
igl::AABB<MatrixXd, 3> tree_template;

// scanned data
MatrixXd V_scanned;
MatrixXd N_scanned;
MatrixXi F_scanned;
VectorXi landmarks_scanned;
MatrixXd landmarks_scanned_points;
igl::AABB<MatrixXd, 3> tree_scanned;

// other data structures
VectorXi boundary_indices;
igl::AABB<MatrixXd, 3> boundary_tree;

// hyperparameters
double lambda = 2.5;
double threshold_distance_percentage = 0.8;
double threshold_parallel_angle_tolerance = 0.6;

// constraints
SparseMatrix<double> constraint_matrix_static;
SparseMatrix<double> constraint_matrix_dynamic;
MatrixXd constraint_rhs_static;
MatrixXd constraint_rhs_dynamic;
bool constraints_computed = false;
int initial_constraint_count = -1;

// other stuff
int iteration_count = 0;

void iterate() {
    // one warping iteration
    if (initial_constraint_count == -1) {
        cout << "Initial constraints have not been computed yet." << endl;
        return;
    }

    if (!constraints_computed) {
        cout << "Constraints for iteration " << iteration_count << " have not been computed yet." << endl;
        return;
    }

    // compute Laplacian
    SparseMatrix<double> laplacian;
    igl::cotmatrix(V_template, F_template, laplacian);

    // concatenate with constraint matrices
    SparseMatrix<double> temp1, constraint_matrix;
    igl::cat(1, laplacian, constraint_matrix_static, temp1);
    igl::cat(1, temp1, constraint_matrix_dynamic, constraint_matrix);
    SparseQR<SparseMatrix<double>, COLAMDOrdering<int>> solver;
    solver.compute(constraint_matrix);

    // compute solution separately for each dimension
    VectorXd temp2, constraint_rhs;
    MatrixXd solution(V_template.rows(), 3);
    for (int dim = 0; dim < 3; dim++) {
        igl::cat(1, VectorXd(laplacian * V_template.col(dim)), VectorXd(constraint_rhs_static.col(dim)), temp2);
        igl::cat(1, temp2, VectorXd(constraint_rhs_dynamic.col(dim)), constraint_rhs);
        V_template.col(dim) = solver.solve(constraint_rhs);
    }

    constraints_computed = false;
    iteration_count++;
}

void compute_constraints() {
    // compute constraints for points not on the boundary or the landmarks

    // compute average distance of vertices in the template to the scanned face
    VectorXd squared_distances;
    VectorXi face_indices;
    MatrixXd closest_points;
    tree_scanned.squared_distance(V_scanned, F_scanned, V_template, squared_distances, face_indices, closest_points);
    double average_distance = squared_distances.array().sqrt().mean();

    // also compute distances to boundary
    VectorXd boundary_distances;
    VectorXi temp1;
    MatrixXd temp2;
    boundary_tree.squared_distance(V_template, boundary_indices, V_template, boundary_distances, temp1, temp2);
    boundary_distances = boundary_distances.array().sqrt().matrix();

    // add constraint for each template vertex that fulfills the requirements
    // would be best to resize constraint matrices here already...
    Vector3d vertex_template, normal_template, normal_scanned, closest_point_scanned, diff_scanned_template;
    int constraint_count = initial_constraint_count, dynamic_constraint_count = 0;
    constraint_matrix_dynamic.setZero();
    vector<Triplet<double>> constraint_matrix_coefficients;
    vector<double> constraint_rhs_list;
    for (int i = 0; i < V_template.rows(); i++) {
        // if the vertex is on the boundary a constraint already exists
        if ((boundary_indices.array() == i).any()) {
            continue;
        }

        // assign or compute the necessary points and vectors
        vertex_template = V_template.row(i);
        normal_template = N_template.row(i);
        normal_scanned = N_scanned.row(face_indices(i)); // NOTE: these are per-face normals!
        closest_point_scanned = closest_points.row(i);
        diff_scanned_template = closest_point_scanned - vertex_template;

        // if the template vertex is too far away from the mesh, it should just be unconstrained
        if (diff_scanned_template.norm() > average_distance * threshold_distance_percentage) {
            continue;
        }

        // if the normals of the template vertex and the closest point in different directions the vertex is unconstrained
        if (normal_template.dot(normal_scanned) < threshold_parallel_angle_tolerance) { 
            // TODO: need to check whether the normals are already normalized
            continue;
        }

        // similarly, if the template vertex normal points in a different direction to the difference vector, skip this vertex
        diff_scanned_template.normalize();
        if (abs(diff_scanned_template.dot(normal_template)) < 0.5) {
            continue;
        }

        double sigma = 2.5; // not sure how they got this number
        double distance_to_boundary = boundary_distances(i);
        distance_to_boundary = (1 / (1 + exp(-(distance_to_boundary - sigma) * (6 / sigma))));
        constraint_matrix_coefficients.push_back(Triplet<double>(constraint_count, i, pow(normal_template.dot(normal_scanned), 10) * distance_to_boundary * lambda));
        constraint_rhs_list.push_back(closest_point_scanned(0) * lambda);
        constraint_rhs_list.push_back(closest_point_scanned(1) * lambda);
        constraint_rhs_list.push_back(closest_point_scanned(2) * lambda);
        constraint_count++;
        dynamic_constraint_count++;
    }
    constraint_matrix_dynamic.setFromTriplets(constraint_matrix_coefficients.begin(), constraint_matrix_coefficients.end());
    constraint_rhs_dynamic = Map<MatrixXd>(constraint_rhs_list.data(), dynamic_constraint_count, 3);
    
    constraints_computed = true;
}



void compute_initial_constraints() {
    // compute the constraints for landmarked points and the boundary
    if (landmarks_template.size() != landmarks_scanned.size()) {
        throw "Number of landmarks must be the same for template and scanned face!";
    }
    int constraint_count = 0;

    // compute constraints for landmarked points
    // need to read landmark file for template and scanned face for this => which format should be used?
    // depends on how landmarks are represented, would probably be easiest to just use vertex indices
    constraint_matrix_static.resize(landmarks_template.size(), V_template.rows());
    constraint_rhs_static.resize(landmarks_template.size(), 3);
    for (int i = 0; i < landmarks_template.size(); i++) {
        // add constraint with weight 1 => less than for boundary but more than for rest of face
        constraint_matrix_static.insert(constraint_count, landmarks_template(i)) = 1.0 * lambda;
        constraint_rhs_static.row(constraint_count) = V_scanned.row(landmarks_scanned(i)) * lambda;
        constraint_count++;
    }

    // compute constraints for the boundary
    vector<vector<DenseIndex>> boundary_loops;
    igl::boundary_loop(F_template, boundary_loops);
    boundary_indices.resize(boundary_loops[0].size());
    constraint_matrix_static.conservativeResize(constraint_matrix_static.rows() + boundary_indices.size(), V_template.rows());
    constraint_rhs_static.conservativeResize(constraint_matrix_static.rows(), 3);
    MatrixXi boundary_faces(boundary_indices.rows(), 3);
    for (int i = 0; i < boundary_loops[0].size(); i++) {
        boundary_indices(i) = (int) boundary_loops[0][i];
        // add constraint with large weight of 10 => boundary should roughly stay the same
        constraint_matrix_static.insert(constraint_count, boundary_indices(i)) = 10.0 * lambda;
        constraint_rhs_static.row(constraint_count) = V_template.row(boundary_indices(i)) * lambda;
        constraint_count++;
        boundary_faces.row(i) = Vector3i(i, i, i);
    }

    // also construct a data structure for distance computations to the boundary
    boundary_tree.init(igl::slice(V_template, boundary_indices), boundary_faces);

    initial_constraint_count = constraint_count;
}

void init() {
    tree_template.init(V_template, F_template);
    tree_scanned.init(V_scanned, F_scanned);

    igl::per_face_normals(V_scanned, F_scanned, Vector3d(1, 1, 1).normalized(), N_scanned);
}

bool callback_key_down(Viewer &viewer, unsigned char key, int modifiers) {
    if (key == 'I') {
        if (initial_constraint_count == -1) {
            compute_initial_constraints();
        }
        compute_constraints();
        iterate();
    }

    return true;
}

void readLandmark(string fileName, MatrixXd &points, const MatrixXd &V_) {
    std::ifstream landfile(fileName);
    int v1, v2, v3;
    float alpha, beta, gamma;
    while (landfile >> v1 >> v2 >> v3 >> alpha >> beta >> gamma) {
        points.conservativeResize(points.rows() + 1, 3);
        points.row(points.rows() -1) =  V_.row(v1)*alpha + V_.row(v2)*beta + V_.row(v3)*gamma;
    }
}

int main(int argc, char *argv[]) {
    string file_name;
    if (argc != 2) {
        cout << "Usage: alignment <scanned_face>" << endl;
        return 0;
    } 

    // load data and initialize everything
    // TODO: also need to load landmarks
    MatrixXd temp1;
    MatrixXi temp2, temp3;
    igl::readOBJ("../data/template.obj", V_template, temp1, N_template, F_template, temp2, temp3);
    readLandmark("../data/template.mark", landmarks_template_points, V_template);

    file_name = argv[1];
    igl::readOBJ( file_name + ".obj", V_scanned, temp1, N_scanned, F_scanned, temp2, temp3);
    readLandmark( file_name + ".mark", landmarks_scanned_points, V_scanned);
    
    Viewer viewer;
    viewer.data().clear();
    viewer.data().set_mesh(V_template, F_template);
    viewer.core.align_camera_center(V_template);

    
    size_t index = file_name.find_last_of("/");
    file_name = file_name.substr(index + 1);
    init();

    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    viewer.callback_key_down = callback_key_down;

    menu.callback_draw_viewer_menu = [&]() {
        // draw parent menu content
        menu.draw_viewer_menu();

        // add new group
        if (ImGui::CollapsingHeader("Warping options", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::InputDouble("Lambda", &lambda, 0, 0);

            if (ImGui::Button("Save mesh", ImVec2(-1, 0))) {
                igl::writeOBJ("../results/" + file_name, V_template, F_template);
            }
        }
    };

    viewer.launch();
}

