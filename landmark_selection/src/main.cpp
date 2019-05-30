#include <igl/readOBJ.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <iostream>
#include <igl/unproject_onto_mesh.h>
#include <string> 
#include <igl/png/render_to_png.h>
#include <igl/png/writePNG.h>

using namespace Eigen;
using namespace std;

typedef igl::opengl::glfw::Viewer Viewer;

bool callback_pre_draw(Viewer& viewer);
void writeOut(string placeToBe);

struct Landmark {
    int v1, v2, v3;
    float alpha, beta, gamma;
};

MatrixXd V;
MatrixXi F;

bool selecting = false;
vector<Landmark> landmarks;

bool mouse_down(Viewer& viewer, int button, int modifier) {
    double x = viewer.current_mouse_x;
    double y = viewer.core.viewport(3) - viewer.current_mouse_y;

    if (selecting) {
        Eigen::Vector3f baryC;
        int fid;     
        if (igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core.view,
            viewer.core.proj, viewer.core.viewport, V, F, fid, baryC)) {
            
            int v1 = F(fid, 0), v2 = F(fid, 1), v3 = F(fid, 2);
            float alpha = baryC[0], beta = baryC[1], gamma = baryC[2];
            
            Landmark mark = {v1, v2, v3, alpha, beta, gamma};
            landmarks.push_back(mark);

            // max bary coords, get nearearst vertex
            long c; baryC.maxCoeff(&c);
            
            RowVector3d nn_vertex = V.row(F(fid,c));
            return true;
        }
    }

    return false;
}

bool replace(string& str, const string& from, const string& to) {
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

int main(int argc, char *argv[]) {
    string faceName;
    if(argc == 1) {
        cout << "---------------------------" << endl;
        cout << "Usage <bin> face.obj" << endl;
        cout << "---------------------------" << endl;
        return 0;
    } else {
        faceName = string(argv[1]);
        igl::readOBJ(faceName, V, F);
    }

    Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);
    viewer.core.align_camera_center(V);

    // Draw additional windows
    menu.callback_draw_custom_window = [&]() {
        ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiSetCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(200, 160), ImGuiSetCond_FirstUseEver);
        ImGui::Begin( "Selection Window", nullptr, ImGuiWindowFlags_NoSavedSettings );

        ImGui::Checkbox("Start Selecting", &selecting);
        if (ImGui::Button("Remove last selected"))  {
            if (landmarks.size() > 0)
                landmarks.pop_back();
        }

        if (ImGui::Button("Update Labels"))  {
            // clear labels
            viewer.data().labels_positions.resize(0, 3);
            viewer.data().labels_strings.clear();

            for (int i = 0; i < landmarks.size(); i++) {
                Landmark mark = landmarks[i];
                RowVector3d point = V.row(mark.v1) * mark.alpha + V.row(mark.v2) * mark.beta + V.row(mark.v3) * mark.gamma;
                string ctr = std::to_string(i);
                viewer.data().add_label(point, ctr);    
            }

            
        }

        if (ImGui::Button("Save"))  {
            writeOut(faceName);
            string copyf = faceName;
            replace(copyf, ".obj", ".png");

            // Allocate temporary buffers
            Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R(1280,800);
            Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> G(1280,800);
            Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> B(1280,800);
            Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> A(1280,800);

            // Draw the scene in the buffers
            viewer.core.draw_buffer(viewer.data(),false,R,G,B,A);

            // Save it to a PNG
            igl::png::writePNG(R,G,B,A, copyf);
        }

        ImGui::End();
    };

    viewer.callback_mouse_down = &mouse_down;
    viewer.callback_pre_draw = callback_pre_draw;

    viewer.data().set_mesh(V, F);
    viewer.launch();
}

bool callback_pre_draw(Viewer& viewer) {
    // clear points
    viewer.data().set_points(MatrixXd::Zero(0,3), MatrixXd::Zero(0,3));

    for (Landmark mark : landmarks) {
        RowVector3d point = V.row(mark.v1)*mark.alpha + V.row(mark.v2)*mark.beta + V.row(mark.v3)*mark.gamma;
        viewer.data().add_points(point, Eigen::RowVector3d(0.0,0.5,0.3));
    }
    return false;
}

void writeOut(string placeToBe) {
    replace(placeToBe, ".obj", ".mark");
    cout << "Save To: " << placeToBe  << endl;

    ofstream s(placeToBe);
    if(!s.is_open()) {
        fprintf(stderr,"IOError: writeOut() could not open %s\n", placeToBe.c_str());
        return;
    }

    for (int i = 0; i < landmarks.size(); i++) {
        Landmark m = landmarks[i];
        s << i << " " << m.v1 << " " << m.v2 << " " << m.v3 << " " << m.alpha << " " << m.beta << " " << m.gamma << "\n";
    }
}