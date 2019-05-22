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
VectorXi boundary_indices;
igl::AABB<MatrixXd, 3> boundary_tree;

// hyperparameters
double lambda = 0.05;
double threshold_distance_percentage = 0.8;
double threshold_parallel_angle_tolerance = 0.6;
bool use_scanned_boundary = false;

// constraints
SparseMatrix<double> constraint_matrix_static;
SparseMatrix<double> constraint_matrix_dynamic;
MatrixXd constraint_rhs_static;
MatrixXd constraint_rhs_dynamic;
bool aligned = false;
bool constraints_computed = false;
int initial_constraint_count = -1;

enum MESH_VIEW { VIEW_SCANNED, VIEW_TEMPLATE, VIEW_BOTH};
MESH_VIEW active_view = VIEW_BOTH;

// other stuff
int iteration_count = 0;

void display_two_meshes(MatrixXd &V1, MatrixXi &F1, MatrixXd &V2, MatrixXi &F2) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd C;

    Eigen::MatrixXd point_template(landmarks_template.size(), 3);
    Eigen::MatrixXd point_scanned(landmarks_scanned.size(), 3);

    for (int i = 0; i < landmarks_scanned.size(); ++i) {
        point_template.row(i) = V1.row(landmarks_template(i));
        point_scanned.row(i) = V2.row(landmarks_scanned(i));
    }

    if (active_view == VIEW_TEMPLATE) {
        V = V1;
        F = F1;
        C.resize(F.rows(), 3);
        C << Eigen::RowVector3d(0.2, 0.3, 0.8).replicate(F.rows(), 1);
    } else if (active_view == VIEW_SCANNED) {
        V = V2;
        F = F2;
        C.resize(F.rows(), 3);
        C << Eigen::RowVector3d(1.0, 0.7, 0.2).replicate(F.rows(), 1);
    } else {
        V.resize(V1.rows() + V2.rows(), V1.cols());
        V << V1, V2;
        F.resize(F1.rows() + F2.rows(), F1.cols());
        F << F1, (F2.array() + V1.rows());

        C.resize(F.rows(), 3);
        C << Eigen::RowVector3d(0.2, 0.3, 0.8).replicate(F1.rows(), 1), Eigen::RowVector3d(1.0, 0.7, 0.2).replicate(F2.rows(), 1);
    }

	viewer.data().clear();
	viewer.data().set_mesh(V, F);
	viewer.data().set_colors(C);
	viewer.data().set_face_based(true);
	viewer.core.align_camera_center(V1);


    if (active_view == VIEW_BOTH || active_view == VIEW_SCANNED) {
        viewer.data().add_points(point_scanned, Eigen::RowVector3d(0.0, 1.0, 0.0).replicate(point_scanned.rows(), 1));
    }

    if (active_view == VIEW_BOTH || active_view == VIEW_TEMPLATE) {
        viewer.data().add_points(point_template, Eigen::RowVector3d(1.0, 0.0, 0.0).replicate(point_template.rows(), 1));
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

    double scaling_factor = mean_distance_scanned / mean_distance_template;

    V_template = V_template * scaling_factor;
    landmarks_template_points = landmarks_template_points * scaling_factor;

    // do rigid alignment using landmark points
    // compute "covariance" matrix of landmarks
    MatrixXd covariance_matrix = landmarks_template_points.transpose() * landmarks_scanned_points; // points should be column vectors here

    // compute SVD and rotation matrix
    JacobiSVD<MatrixXd> svd(covariance_matrix, ComputeFullU | ComputeFullV);
    MatrixXd optimal_rotation_matrix = svd.matrixV() * svd.matrixU().transpose();

    // compute rotated points
    V_template = (optimal_rotation_matrix * V_template.transpose()).transpose();
    cout << "V_template shape: " << V_template.rows() << "x" << V_template.cols() << endl;
    cout << V_template.row(0) << endl;

    // most of this can be commented out/removed, it's just to check whether the faces align somewhat reasonably
    Eigen::MatrixXd V(V_template.rows() + V_scanned.rows(), V_template.cols());
    V << V_template, V_scanned;
    Eigen::MatrixXi F(F_template.rows() + F_scanned.rows(), F_template.cols());
    F<< F_template, (F_scanned.array() + V_template.rows());

    Eigen::MatrixXd C(F.rows(), 3);
    C << Eigen::RowVector3d(0.2, 0.3, 0.8).replicate(F_template.rows(), 1),  Eigen::RowVector3d(1.0, 0.7, 0.2).replicate(F_scanned.rows(), 1);

    display_two_meshes(V_template, F_template, V_scanned, F_scanned);

    aligned = true;
}

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

    igl::per_face_normals(V_template, F_template, Vector3d(1, 1, 1).normalized(), N_template);
    igl::per_face_normals(V_scanned, F_scanned, Vector3d(1, 1, 1).normalized(), N_scanned);
}


MatrixXi F_boundary;
MatrixXd V_boundary;

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
        normal_scanned = N_scanned.row(face_indices(i)); // NOTE: these are per-face normals!
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

        double sigma = 2.5; // not sure how they got this number
        double distance_to_boundary = sqrt(boundary_distances(i));
        distance_to_boundary = (1.0 / (1.0 + exp(-(distance_to_boundary - sigma) * (6.0 / sigma))));

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
            if (!use_scanned_boundary) {
                // use original point on the template
                rhs_fixed.row(t_i) = c_weight * 1.0 * V.row(loop[i]);
            } else {
                // use closest point on scanned face boundary
                int min;
                (V_target_boundary.rowwise() - V.row(loop[i])).rowwise().norm().minCoeff(&min);
                rhs_fixed.row(t_i) = c_weight * 1.0 * V_target_boundary.row(min);
            }
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
    Eigen::SparseMatrix<double> Afull;
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
    if (key == 'I') {
        warping_step();
        return true;

        // Previous version
        if (initial_constraint_count == -1) {
            compute_initial_constraints();
        }
        compute_constraints();
        iterate();
    }

    if (key == 'R') {
        rigid_align();
    }

    if (key >= '1' && key <= '3') {
        switch (key)
        {
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
        cout << "Usage: alignment <scanned_face>" << endl;
        file_name = "../data/simonw_neutral_corrected";
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

    display_two_meshes(V_template, F_template, V_scanned, F_scanned);

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

            ImGui::Checkbox("Use scanned boundary", &use_scanned_boundary);

            if (ImGui::Button("Rigidly align", ImVec2(-1, 0))) {
                rigid_align();
            }

            if (ImGui::Button("Warping step", ImVec2(-1, 0))) {
                warping_step();
            }

            if (ImGui::Button("Save mesh", ImVec2(-1, 0))) {
                igl::writeOBJ(file_name + "_warped.obj", V_template, F_template);
            }
        }
    };

    viewer.launch();
}
