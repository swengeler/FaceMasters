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

Viewer viewer;

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
MatrixXi F_boundary;
MatrixXd V_boundary;
VectorXi boundary_indices;
igl::AABB<MatrixXd, 3> boundary_tree;

// hyperparameters and UI options
double lambda = 0.05;
double sigma = 5;
double threshold_distance_percentage = 0.8;
double threshold_parallel_angle_tolerance = 0.6;

// constraints
SparseMatrix<double> constraint_matrix_static;
SparseMatrix<double> constraint_matrix_dynamic;
MatrixXd constraint_rhs_static;
MatrixXd constraint_rhs_dynamic;
bool aligned = false;
bool constraints_computed = false;
int initial_constraint_count = -1;

// rigid alignment information
MatrixXd optimal_rotation_matrix;
double scaling_factor;

enum MESH_VIEW { VIEW_SCANNED, VIEW_TEMPLATE, VIEW_BOTH};
MESH_VIEW active_view = VIEW_BOTH;

void display_two_meshes(MatrixXd &V1, MatrixXi &F1, MatrixXd &V2, MatrixXi &F2) {
    MatrixXd V;
    MatrixXi F;
    MatrixXd C;

    MatrixXd point_template(landmarks_template.size(), 3);
    MatrixXd point_scanned(landmarks_scanned.size(), 3);

    for (int i = 0; i < landmarks_scanned.size(); ++i) {
        point_template.row(i) = V1.row(landmarks_template(i));
        point_scanned.row(i) = V2.row(landmarks_scanned(i));
    }

    if (active_view == VIEW_TEMPLATE) {
        V = V1;
        F = F1;

        C.resize(F.rows(), 3);
        C << RowVector3d(0.2, 0.3, 0.8).replicate(F.rows(), 1);
    } else if (active_view == VIEW_SCANNED) {
        V = V2;
        F = F2;
        C.resize(F.rows(), 3);
        C << RowVector3d(1.0, 0.7, 0.2).replicate(F.rows(), 1);
    } else {
        V.resize(V1.rows() + V2.rows(), V1.cols());
        V << V1, V2;
        F.resize(F1.rows() + F2.rows(), F1.cols());
        F << F1, (F2.array() + V1.rows());

        C.resize(F.rows(), 3);
        C << RowVector3d(0.2, 0.3, 0.8).replicate(F1.rows(), 1), RowVector3d(1.0, 0.7, 0.2).replicate(F2.rows(), 1);
    }

	viewer.data().clear();
	viewer.data().set_mesh(V, F);
	viewer.data().set_colors(C);
	viewer.data().set_face_based(true);
	viewer.core.align_camera_center(V1);

    if (active_view == VIEW_BOTH || active_view == VIEW_SCANNED) {
        viewer.data().add_points(point_scanned, RowVector3d(0.0, 1.0, 0.0).replicate(point_scanned.rows(), 1));
    }

    if (active_view == VIEW_BOTH || active_view == VIEW_TEMPLATE) {
        viewer.data().add_points(point_template, RowVector3d(1.0, 0.0, 0.0).replicate(point_template.rows(), 1));
    }
}

void rigid_align() {
    // move template and scanned face to center
    RowVector3d mean_template = V_template.colwise().mean();
    V_template = V_template.rowwise() - mean_template;
    landmarks_template_points = landmarks_template_points.rowwise() - mean_template;

    RowVector3d mean_scanned = V_scanned.colwise().mean();
    V_scanned = V_scanned.rowwise() - mean_scanned;
    landmarks_scanned_points = landmarks_scanned_points.rowwise() - mean_scanned;

    // scale the template
    RowVector3d mean_landmark_template = landmarks_template_points.colwise().mean();
    double mean_distance_template = (landmarks_template_points.rowwise() - mean_landmark_template).rowwise().norm().mean();

    RowVector3d mean_landmark_scanned = landmarks_scanned_points.colwise().mean();
    double mean_distance_scanned = (landmarks_scanned_points.rowwise() - mean_landmark_scanned).rowwise().norm().mean();

    scaling_factor = mean_distance_scanned / mean_distance_template;

    V_template = V_template * scaling_factor;
    landmarks_template_points = landmarks_template_points * scaling_factor;

    // do rigid alignment using landmark points
    // compute "covariance" matrix of landmarks
    MatrixXd covariance_matrix = landmarks_template_points.transpose() * landmarks_scanned_points; // points should be column vectors here

    // compute SVD and rotation matrix
    JacobiSVD<MatrixXd> svd(covariance_matrix, ComputeFullU | ComputeFullV);
    optimal_rotation_matrix = svd.matrixV() * svd.matrixU().transpose();

    // compute rotated points
    V_template = (optimal_rotation_matrix * V_template.transpose()).transpose();

    display_two_meshes(V_template, F_template, V_scanned, F_scanned);

    aligned = true;
}

void init() {
    tree_template.init(V_template, F_template);
    tree_scanned.init(V_scanned, F_scanned);

    igl::per_face_normals(V_template, F_template, Vector3d(1, 1, 1).normalized(), N_template);
    igl::per_face_normals(V_scanned, F_scanned, Vector3d(1, 1, 1).normalized(), N_scanned);
}

void build_dynamic_constraints(SparseMatrix<double> &A, MatrixXd &rhs) {
    // These two should be precomputed after alignment
    tree_scanned.init(V_scanned, F_scanned);
    igl::per_face_normals(V_scanned, F_scanned, Vector3d(1, 1, 1).normalized(), N_scanned);
    igl::per_face_normals(V_template, F_template, Vector3d(1, 1, 1).normalized(), N_template);

    // compute average distance of vertices in the template to the scanned face
    VectorXd squared_distances;
    VectorXi face_indices;
    MatrixXd closest_points;
    tree_scanned.squared_distance(V_scanned, F_scanned, V_template, squared_distances, face_indices, closest_points);

    // also compute distances to boundary
    VectorXd boundary_distances;
    VectorXi temp1;
    MatrixXd temp2;
    boundary_tree.squared_distance(V_boundary, F_boundary, V_template, boundary_distances, temp1, temp2);

    double average_distance = squared_distances.array().sqrt().mean();

    // add constraint for each template vertex that fulfills the requirements
    // would be best to resize constraint matrices here already...
    Vector3d vertex_template, normal_template, normal_scanned, closest_point_scanned, diff_scanned_template;
    int constraint_count = 0;

    vector<Triplet<double>> lhs_vector;
    vector<RowVector3d> rhs_vector;
    for (int i = 0; i < V_template.rows(); i++) {
        // if the vertex is on the boundary a constraint already exists
        if ((boundary_indices.array() == i).any()) {
            continue;
        }

        // if the vertex is landmark, also ignore
        if ((landmarks_template.array() == i).any()) {
            continue;
        }

        // assign or compute the necessary points and vectors
        vertex_template = V_template.row(i);
        normal_template = N_template.row(i);
        normal_scanned = N_scanned.row(face_indices(i));
        closest_point_scanned = closest_points.row(i);
        diff_scanned_template = closest_point_scanned - vertex_template;

        // if the template vertex is too far away from the mesh, it should just be unconstrained
        if (diff_scanned_template.norm() > average_distance * threshold_distance_percentage) {
            continue;
        }

        // if the normals of the template vertex and the closest point in different directions the vertex is unconstrained
        if (normal_template.dot(normal_scanned) < threshold_parallel_angle_tolerance) {
            continue;
        }

        // similarly, if the template vertex normal points in a different direction to the difference vector, skip this vertex
        diff_scanned_template.normalize();
        if (abs(diff_scanned_template.dot(normal_template)) < 0.5) {
            continue;
        }

        double scaled_sigma = sigma * scaling_factor;
        double distance_to_boundary = sqrt(boundary_distances(i));
        distance_to_boundary = (1.0 / (1.0 + exp(-(distance_to_boundary - scaled_sigma) * (6.0 / scaled_sigma))));

        double weight = pow(normal_template.dot(normal_scanned), 10) * distance_to_boundary * lambda;

        lhs_vector.push_back(Triplet<double>(constraint_count, i, weight));
        rhs_vector.push_back(closest_point_scanned *  weight);
        constraint_count++;
    }

    A.resize(constraint_count, V_template.rows());
    A.setZero();
    A.setFromTriplets(lhs_vector.begin(), lhs_vector.end());

    rhs.resize(constraint_count, 3);
    for (int i = 0; i < constraint_count; i++) {
        rhs.row(i) = rhs_vector[i];
    }
}

void warping_step() {
    if (landmarks_template.size() != landmarks_scanned.size()) {
        throw "Number of landmarks must be the same for template and scanned face!";
    }

    double c_weight = lambda * 10;

    MatrixXd V = V_template;
    MatrixXd V_target = V_scanned;
    SparseMatrix<double> C_full;
    MatrixXd C_full_rhs;
    MatrixXd rhs_fixed;
    SparseMatrix<double> C;
    MatrixXd rhs;

    SimplicialCholesky<SparseMatrix<double>> solver;
    SparseMatrix<double> laplacian;

    // Boundary and landmark contraints
    {
        vector<vector<int>> boundary_loops;
        igl::boundary_loop(F_template, boundary_loops);
        vector<int> loop = boundary_loops[0];

        C.resize(landmarks_template.size() + loop.size(), V.rows());
        rhs_fixed.resize(landmarks_template.size() + loop.size(), 3);

        for (int i = 0; i < landmarks_template.size(); ++i) {
            C.insert(i, landmarks_template(i)) = c_weight;
            rhs_fixed.row(i) = c_weight * V_scanned.row(landmarks_scanned(i));
        }

        boundary_indices.resize(loop.size());

        F_boundary.resize(loop.size(), 3);

        // get boundary loop of scanned vertices
        igl::boundary_loop(F_scanned, boundary_loops);
        vector<int> scanned_loop = boundary_loops[0];
        MatrixXd V_target_boundary(scanned_loop.size(), V_scanned.cols());
        for (int i = 0; i < scanned_loop.size(); i++) {
            V_target_boundary.row(i) = V_scanned.row(scanned_loop[i]);
        }

        for (int i = 0; i < loop.size(); ++i) {
            boundary_indices(i) = loop[i];
            F_boundary.row(i) = RowVector3i(i, i, i);

            int t_i = i + landmarks_template.size();
            C.insert(t_i, loop[i]) = c_weight * 1.0;
            rhs_fixed.row(t_i) = c_weight * 1.0 * V.row(loop[i]);
        }

        igl::slice(V_template, boundary_indices, 1, V_boundary);

        boundary_tree.init(V_boundary, F_boundary);
    }

    // Build dynamic constants from closest point on mesh
    SparseMatrix<double> C_dynamic;
    MatrixXd C_dynamic_rhs;
    build_dynamic_constraints(C_dynamic, C_dynamic_rhs);

    // Add contraints together
    igl::cat(1, C, C_dynamic, C_full);
    igl::cat(1, rhs_fixed, C_dynamic_rhs, C_full_rhs);

    // Laplacian part of linear system
    igl::cotmatrix(V, F_template, laplacian);
    laplacian = -laplacian;
    rhs = laplacian * V;

    // Add together linear system
    SparseMatrix<double> Afull;
    igl::cat(1, laplacian, C_full, Afull);
    MatrixXd rhs_full;
    igl::cat(1, rhs, C_full_rhs, rhs_full);

    // Solve
    rhs_full = Afull.transpose() * rhs_full;
    Afull = Afull.transpose() * Afull;
    Afull.makeCompressed();
    solver.compute(Afull);
    V_template = solver.solve(rhs_full);

    // Store and show mesh
    V_template.conservativeResize(laplacian.rows(), 3);
    display_two_meshes(V_template, F_template, V_scanned, F_scanned);
}

void readLandmark(string fileName, MatrixXd &points, VectorXi &indices, const MatrixXd &V_) {
    ifstream landfile(fileName);
    int index, v1, v2, v3;
    float alpha, beta, gamma;
    string line;
    while (getline(landfile, line)) {
        stringstream line_stream(line);
        line_stream >> index >> v1 >> v2 >> v3 >> alpha >> beta >> gamma;
        points.conservativeResize(points.rows() + 1, 3);
        points.row(points.rows() - 1) = V_.row(v1) * alpha + V_.row(v2) * beta + V_.row(v3) * gamma;

        // Choose max for landmark vertex
        indices.conservativeResize(indices.size() + 1);
        float max_v = max(alpha, max(beta, gamma)) - 1e-5;
        if (alpha > max_v) {
          indices(indices.size() - 1) = v1;
        } else if (beta > max_v) {
          indices(indices.size() - 1) = v2;
        } else {
          indices(indices.size() - 1) = v3;
        }
    }
}

bool callback_key_down(Viewer &viewer, unsigned char key, int modifiers) {
    if (key == 'R') {
        rigid_align();
    }

    if (key == 'I') {
        warping_step();
    }

    if (key >= '1' && key <= '3') {
        switch (key) {
            case '1':
                active_view = VIEW_SCANNED;
                break;
            case '2':
                active_view = VIEW_TEMPLATE;
                break;
            case '3':
                active_view = VIEW_BOTH;
                break;
        }
        display_two_meshes(V_template, F_template, V_scanned, F_scanned);
    }

    return true;
}

int main(int argc, char *argv[]) {
    string file_name;
    if (argc != 2) {
        cout << "-------------------------------------------" << endl;
        cout << "Usage: FaceMasters-Alignment <scanned_face>" << endl;
        cout << "-------------------------------------------" << endl;
        return 0;
    } else {
        file_name = argv[1];
    }

    // load data and initialize everything
    size_t index = file_name.find_last_of("/");
    string directory_name = file_name.substr(0, index + 1);

    MatrixXd temp1;
    MatrixXi temp2, temp3;
    igl::readOBJ(directory_name + "template.obj", V_template, temp1, N_template, F_template, temp2, temp3);
    readLandmark(directory_name + "template.mark", landmarks_template_points, landmarks_template, V_template);

    igl::readOBJ(file_name + ".obj", V_scanned, temp1, N_scanned, F_scanned, temp2, temp3);
    readLandmark(file_name + ".mark", landmarks_scanned_points, landmarks_scanned, V_scanned);

    init();
    display_two_meshes(V_template, F_template, V_scanned, F_scanned);

    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);
    viewer.callback_key_down = callback_key_down;

    menu.callback_draw_viewer_menu = [&]() {
        // draw parent menu content
        menu.draw_viewer_menu();

        // add new group
        if (ImGui::CollapsingHeader("Warping options", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::InputDouble("Lambda", &lambda, 0, 0);
            ImGui::InputDouble("Sigma", &sigma, 0, 0);

            if (ImGui::Button("Rigidly align", ImVec2(-1, 0))) {
                rigid_align();
            }

            if (ImGui::Button("Warping step", ImVec2(-1, 0))) {
                warping_step();
            }

            if (ImGui::Button("Save mesh", ImVec2(-1, 0))) {
                MatrixXd V_output = (V_template.rowwise() - V_template.colwise().mean()) * optimal_rotation_matrix.inverse().transpose();
                V_output /= scaling_factor;
                igl::writeOBJ(file_name + "_warped.obj", V_output, F_template);
            }
        }
    };

    viewer.data().show_lines = false;
    viewer.launch();
}
