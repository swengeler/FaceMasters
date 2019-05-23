#include <igl/readOBJ.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <iostream>
#include <igl/unproject_onto_mesh.h>
#include <string>
#include <igl/is_file.h>

using namespace Eigen;
using namespace std;


typedef igl::opengl::glfw::Viewer Viewer;
Viewer viewer;

bool callback_pre_draw(Viewer& viewer);

// each column is a face
MatrixXd V_Faces(0,0);
MatrixXi F;

// solver for PCA
EigenSolver<MatrixXd> eig;

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
    
    string names[] = {"ali", "arda", "christian", "karlis", "patrick", "qais", "shanshan", "simonh", "simonw"};

    for (int i = 0; i < 9; i++) {
        string normal = baseDir + names[i] + "_neutral/0_SFusion.obj";
        string smile = baseDir + names[i] + "_smile/0_SFusion.obj";

        if (!igl::is_file(normal.c_str())) {
            cout << "face: " << normal << " not found, quitting loading" << endl;
            break;
        }

        readFaceAndAddToFaces(normal);
        readFaceAndAddToFaces(smile);
    }

    cout << "loaded: " << V_Faces.cols() << " faces" << " with " << V_Faces.rows() << " points each" << endl;
}


void pca() {
    // calculate average face
    VectorXd avg_face = V_Faces.rowwise().mean();
    // subtract average face
    V_Faces.colwise() -= avg_face;

    // calculate eigenvectors and values on covariance matrix of faces
    eig = EigenSolver<MatrixXd>(V_Faces.transpose()*V_Faces);
    eig.eigenvalues();
    eig.eigenvectors();

    // this is the fast way normally XX' would be the cov. matrix so to get to the eigv of the cov matrix
    // we have to multiply it by X again , maybe normalize eigv. if we need them normalized ?
    for (int i = 0; i < eig.eigenvectors().cols(); i++) {
        //VectorXd i_eigenvector = V_Faces*eig.eigenvectors().col(i);
    }

    // compute eigenfaces and average face
    V_Eigenfaces = V_Faces * eig.eigenvectors().real();
    averageFace = Map<MatrixXd>(avg_face.data(), 3, V_Eigenfaces.rows()/3).transpose();
}


void drawComposedFace(){
  MatrixXd composedFace = averageFace;

  for (int i = 0; i < noEigenfaces; i++) {
    MatrixXd eigenface = Map<MatrixXd>(V_Eigenfaces.col(i).data(), 3, V_Eigenfaces.rows()/3).transpose();
    composedFace.noalias() += eigenFaceWeights[i] * eigenface;
  }

  // update visualised data
  viewer.data().set_mesh(composedFace, F);
}


int main(int argc, char *argv[]) {
    string baseDir;
    if(argc == 1) {
        cout << "---------------------------" << endl;
        cout << "Usage <bin> facefolder" << endl;
        cout << "---------------------------" << endl;
        return 0;
    } else {
        baseDir = string(argv[1]);
        readFaces(baseDir);
        pca();
        eigenFaceWeights[0] = 1.0f;
        drawComposedFace();
    }

    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);
    viewer.core.align_camera_center(viewer.data().V);
    viewer.data().show_lines = false;

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

    };

    viewer.callback_pre_draw = callback_pre_draw;
    viewer.launch();
}


bool callback_pre_draw(Viewer& viewer) {
  return false;
}
