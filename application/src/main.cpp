#include <igl/readOBJ.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <iostream>
#include <igl/unproject_onto_mesh.h>
#include <string>
#include <igl/is_file.h>
#include <igl/png/readPNG.h>

using namespace Eigen;
using namespace std;


typedef igl::opengl::glfw::Viewer Viewer;
Viewer viewer;

bool callback_pre_draw(Viewer& viewer);

// each column is a face
MatrixXd V_Faces(0,0);
MatrixXi F;

// solver for PCA
SelfAdjointEigenSolver<MatrixXd> eig;

// number of eigenfaces to use
int noEigenfaces = 5;
// average face (dim: #Vx3)
MatrixXd averageFace;
// eigenfaces stored column-wise
MatrixXd V_Eigenfaces;
// weights of eigenfaces controlled by sliders
vector<float> eigenFaceWeights(noEigenfaces, 0.0f);


void readFaceAndAddToFaces(string face) {
    cout << "loading face: " << face << endl;
    MatrixXd V_tmp;
    MatrixXi F_tmp;
    igl::readOBJ(face, V_tmp, F_tmp);
    // all F_tmp should be the same
    F = F_tmp;
    // eigen stores the values in column by column format
    // so to get row by row we have to transpose first
    V_tmp.transposeInPlace();
    // then flatten the matrix row by row
    VectorXd V_row_temp(Map<VectorXd>(V_tmp.data(), V_tmp.cols()*V_tmp.rows()));
    // and add it as column to the V_Faces
    V_Faces.conservativeResize(V_row_temp.rows(), V_Faces.cols() + 1);
    V_Faces.col(V_Faces.cols() -1) = V_row_temp;
}


void readFaces(string baseDir) {
    cout << "Loading faces from: " << baseDir << endl;

    vector<string> names{"ali", "arda", "christian", "karlis", "patrick", "qais", "shanshan", "simonh", "simonw"};

    for (int i = 0; i < names.size(); i++) {
        string normal = baseDir + names[i] + "_neutral_corrected_warped.obj";
        string smile = baseDir + names[i] + "_smile_corrected_warped.obj";

        if (!igl::is_file(normal.c_str())) {
            cout << "face: " << normal << " not found, skipping" << endl;
            continue;
        }

        readFaceAndAddToFaces(normal);
        readFaceAndAddToFaces(smile);
    }

    cout << "loaded: " << V_Faces.cols() << " faces" << " with " << V_Faces.rows() << " points each" << endl;
    if (noEigenfaces > V_Faces.cols()) {
        throw "More eigenfaces than faces available as input!";
    }
}


void pca() {
    // calculate average face
    VectorXd avg_face = V_Faces.rowwise().mean();
    // subtract average face
    V_Faces.colwise() -= avg_face;

    // calculate eigenvectors and values on covariance matrix of faces
    eig = SelfAdjointEigenSolver<MatrixXd>(V_Faces.transpose()*V_Faces);
    eig.eigenvalues();
    eig.eigenvectors();

    // this is the fast way normally XX' would be the cov. matrix so to get to the eigv of the cov matrix
    // we have to multiply it by X again , maybe normalize eigv. if we need them normalized ?
    for (int i = 0; i < eig.eigenvectors().cols(); i++) {
        //VectorXd i_eigenvector = V_Faces*eig.eigenvectors().col(i);
    }

    // compute eigenfaces and average face
    V_Eigenfaces = V_Faces * eig.eigenvectors().reverse().real();
    averageFace = Map<MatrixXd>(avg_face.data(), 3, V_Eigenfaces.rows()/3).transpose();
}
MatrixXd composedFace;

void drawComposedFace(){
  composedFace = averageFace;

  for (int i = 0; i < noEigenfaces; i++) {
    MatrixXd eigenface = Map<MatrixXd>(V_Eigenfaces.col(i).data(), 3, V_Eigenfaces.rows()/3).transpose();
    composedFace.noalias() += eigenFaceWeights[i] * eigenface;
  }

  // update visualised data

  //viewer.data().clear(); // clear mesh


  viewer.data().set_mesh(composedFace, F);
  viewer.data().compute_normals();

 // FN.setZero(F.rows(), 3);
 // igl::per_face_normals(composedFace, F, FN);
 // viewer.data().set_normals(FN);
 //viewer.core.align_camera_center(composedFace);
}


int main(int argc, char *argv[]) {
    string baseDir;
    Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R,G,B,A;

    if(argc == 1 || argc == 2) {
        cout << "---------------------------" << endl;
        cout << "Usage <bin> facefolder. Also include ../skin.jpg as argument." << endl;
        cout << "---------------------------" << endl;
        return 0;
    } else {
        baseDir = string(argv[1]);
        igl::png::readPNG(argv[2],R,G,B,A);
        readFaces(baseDir);
        pca();
        eigenFaceWeights[0] = 1.0f;
        drawComposedFace();
    }

    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);
    viewer.core.align_camera_center(viewer.data().V);
    //viewer.core.background_color << 253/255.0f, 246/255.0f, 228/255.0f, 1.0f;
    viewer.data().set_texture(R,G,B,A);
    viewer.data().set_face_based(false);
    viewer.data().show_lines = false;
    viewer.data().show_texture = true;
    viewer.launch_init(true,false);

    // Draw additional windows
    menu.callback_draw_viewer_menu = [&]() {

        // menu.draw_viewer_menu(); // draw default menu above

        // Add new group: Sliders for eigenfaces
        if (ImGui::CollapsingHeader("Blending Eigenfaces", ImGuiTreeNodeFlags_DefaultOpen))
        {
          for(int i=0; i<noEigenfaces; i++){
            const string label = "eigenface " + to_string(i);
            if (ImGui::SliderFloat(label.c_str(), &eigenFaceWeights[i], 0.0f, 1.0f)){
              drawComposedFace();
            }
          }
        }

        if (ImGui::Button("Save mesh", ImVec2(-1, 0))) {
            igl::writeOBJ(baseDir + "eigenface_mixture.obj", composedFace, F);
        }

    };

    viewer.callback_pre_draw = callback_pre_draw;

    viewer.data().meshgl.init();
    igl::opengl::destroy_shader_program(viewer.data().meshgl.shader_mesh);
    {
      std::string mesh_vertex_shader_string =
        R"(#version 150
        uniform mat4 view;
        uniform mat4 proj;
        uniform mat4 normal_matrix;
        in vec3 position;
        in vec3 normal;
        out vec3 normal_eye;

        void main()
        {
          normal_eye = normalize(vec3 (normal_matrix * vec4 (normal, 0.0)));
          gl_Position = proj * view * vec4(position, 1.0);
        })";

      std::string mesh_fragment_shader_string =
        R"(#version 150
        in vec3 normal_eye;
        out vec4 outColor;
        uniform sampler2D tex;
        void main()
        {
          vec2 uv = normalize(normal_eye).xy * 0.5 + 0.5;
          outColor = texture(tex, uv);
        })";

      igl::opengl::create_shader_program(mesh_vertex_shader_string,mesh_fragment_shader_string,{},viewer.data().meshgl.shader_mesh);
    }

    viewer.launch_rendering(true);
    viewer.launch_shut();
}


bool callback_pre_draw(Viewer& viewer) {
  return false;
}
