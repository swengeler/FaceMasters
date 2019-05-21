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


bool callback_pre_draw(Viewer& viewer);

// each column is a face
MatrixXd V_Faces(0,0);
MatrixXi F;

void readFaceAndAddToFaces(string face) {
    cout << "loading face: " << face << endl;
    MatrixXd V_tmp;
    MatrixXi F_tmp;
    igl::readOBJ(face, V_tmp, F_tmp);
    // eigen stores the values in column by column format
    // so to get row by row we have to transpose first 
    V_tmp.transposeInPlace();
    // then flatten the matrix row by row 
    VectorXd V_row_temp(Map<VectorXd>(V_tmp.data(), V_tmp.cols()*V_tmp.rows()));
    // and add it as column to the V_faces
    V_Faces.conservativeResize(V_row_temp.rows(), V_Faces.cols() + 1);
    V_Faces.col(V_Faces.cols() -1) = V_row_temp;
}

void readFaces(string baseDir) {
    cout << "Loading faces from: " << baseDir << endl;

    for (int i = 0; i < 20; i++) {
        string normal = baseDir + "/person" + to_string(i+1) + "_normal.obj";
        string smile = baseDir + "/person" + to_string(i+1) + "_smile.obj";
        
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

    // calculate eigenvectos and values on covariance matrix of faces
    EigenSolver<Matrix3d> eig(V_Faces.transpose()*V_Faces); 
    eig.eigenvalues();
    eig.eigenvectors();

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
    }

    Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);


    // Draw additional windows
    menu.callback_draw_custom_window = [&]() {
        ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiSetCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(200, 160), ImGuiSetCond_FirstUseEver);
        ImGui::Begin( "Selection Window", nullptr, ImGuiWindowFlags_NoSavedSettings );


        if (ImGui::Button("Save"))  {
        }

        ImGui::End();
    };

    viewer.callback_pre_draw = callback_pre_draw;

    viewer.launch();
}



bool callback_pre_draw(Viewer& viewer) {

}




