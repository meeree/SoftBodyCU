#define _USE_MATH_DEFINES
#include "graphics.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <fstream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <string>


using namespace std;

namespace Graphics 
{

// Circle buffer struct used for replaying recent frames
struct ReplayBuffer 
{
    std::vector<GLfloat> voltage_buff;
    GLint frame_idx{ 0 }, frame_rewind_idx{ 0 };
    GLint num_frames{ 0 }, frame_size{ 0 };
    
    void init (GLint N, GLint frame_cache_count)
    {
        frame_idx = 0;
        frame_rewind_idx = 0;
        num_frames = frame_cache_count;
        frame_size = N;

        // Allocate frame_cache_count frames for replay
        voltage_buff.resize(N * frame_cache_count);
    }

    void write_frame(float const* voltages)
    {
        frame_rewind_idx = frame_idx; // Set to index of previous frame
        std::copy(voltages, voltages + frame_size, voltage_buff.begin() + frame_idx * frame_size);
        frame_idx = (frame_idx + 1) % num_frames; // Circle around 
    }

    const GLfloat* read_frame() const
    {
        return voltage_buff.data() + frame_rewind_idx * frame_size;
    }

    void move_N_frames(int N) // N can be positive or negative
    {
        // Clamp frame_rewind_idx + N to range [0, frame_idx].
        frame_rewind_idx = min(max(frame_rewind_idx + N, 0), frame_idx);
    }
};

struct 
{
    GLuint vao;
    GLint width, height;
	std::string title;
    GLuint shaderProgram, tex_shaderProgram;
    GLuint texture;
    GLFWwindow* window;
    glm::vec3 cam_pos, cam_forward{ 0, 0, 1 }, cam_right{ 1, 0, 0 };
    glm::vec3 look_pos{ 0,0,0 };
    glm::mat4 view{ glm::mat4(1.0f) }, model{ glm::mat4(1.0f) }, projection{ glm::mat4(1.0f) };
    glm::vec2 mouse{0.0f, 0.0f};
    glm::vec2 theta{0.0f, 0.0f};
    bool pause{ false }, rotate{ false }, replay{ false };
    glm::vec3 world_min, world_max;
    int draw_mode = 0;
    GLfloat color_divisor = 15;
    GLfloat color_offset = -65;
    GLfloat color_cutoff = 0;

    ReplayBuffer replay_buffer; // Used for replaying video 

    // Locations of stuff
    GLint voltage_loc, pos_loc, norm_loc, tex_coord_loc; // attributes 
    GLint p_mat_loc, v_mat_loc, m_mat_loc, sampler_loc, campos_loc; // uniforms
    GLint render_lines_loc, color_divisor_loc, color_offset_loc, color_cutoff_loc;

    GLuint pos_buff, norm_buff, voltage_buff, tex_coord_buff, line_buff, tri_buff;

    int raw_video_frame_idx = 0;
    int max_raw_video_size_gb = 10;
    int mov_video_idx = 0;

    // Application-specific stuff
    vector<GLfloat> voltages;
    vector<glm::vec3> points;
    vector<glm::vec3> normals;
    vector<GLuint> lines;
    vector<GLuint> triangles;
    int slice_index{ 0 };
} g_attrs;

GLuint loadInShader(GLenum const &shaderType, char const *fname) {
   std::vector<char> buffer;
   std::ifstream in;
   in.open(fname, std::ios::binary);

   if (in.is_open()) {
      in.seekg(0, std::ios::end);
      size_t const &length = in.tellg();

      in.seekg(0, std::ios::beg);

      buffer.resize(length + 1);
      in.read(&buffer[0], length);
      in.close();
      buffer[length] = '\0';
   } else {
      std::cerr << "Unable to open " << fname << std::endl;
      exit(-1);
   }

   GLchar const *src = &buffer[0];

   GLuint shader = glCreateShader(shaderType);
   glShaderSource(shader, 1, &src, NULL);
   glCompileShader(shader);
   GLint test;
   glGetShaderiv(shader, GL_COMPILE_STATUS, &test);

   if (!test) {
      std::cerr << "Shader compilation failed with this message:" << std::endl;
      std::vector<char> compilationLog(512);
      glGetShaderInfoLog(shader, compilationLog.size(), NULL, &compilationLog[0]);
      std::cerr << &compilationLog[0] << std::endl;
      glfwTerminate();
      exit(-1);
   }

   return shader;
}

void initShaders (char const* vertLoc, char const* fragLoc, GLuint& shader_prog)
{   
    shader_prog = glCreateProgram();

    auto vertShader = loadInShader(GL_VERTEX_SHADER, vertLoc);
    auto fragShader = loadInShader(GL_FRAGMENT_SHADER, fragLoc);

    glAttachShader(shader_prog, vertShader);
    glAttachShader(shader_prog, fragShader);

    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

    glLinkProgram(shader_prog);
}

void startup (GLfloat const& width, GLfloat const& height, const char* title="Untitled Window")
{   
    g_attrs.width = width; 
    g_attrs.height = height;

    if(!glfwInit()) {
        std::cerr<<"failed to initialize glfw"<<std::endl;
        exit(1);
    }
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	g_attrs.title = title;
    g_attrs.window = glfwCreateWindow(g_attrs.width, g_attrs.height, title, nullptr, nullptr);
    if(!g_attrs.window) {
        std::cerr<<"failed to initialize window"<<std::endl;
        exit(1);
    }
    glfwMakeContextCurrent(g_attrs.window);

    glewExperimental = GL_TRUE;
    if(glewInit() != 0) {
        std::cerr<<"failed to initialize glew"<<std::endl;
        exit(1);
    }
    glGetError(); // We get an INVALID_ENUM error from glew sometimes.
    // It's not a problem. Just wipe the error checking
}

void shutdown () 
{   
    glDeleteVertexArrays(1, &g_attrs.vao);
    glDeleteProgram(g_attrs.shaderProgram);
}

bool trivial_predicate(float const*, float const*) { return true; }

void load_in_connectivity(int const* connectivity_inds, int const* offset_into_inds, int const N, int const N_pts_to_use, LinePredicate pred)
{
    // We use an index buffer to store the indices for the lines. An index buffer is nice 
    // because it means we can use the already loaded positions and just index into them.
    int const stride =  g_attrs.points.size() / N_pts_to_use;
    int const points_stride = N / g_attrs.points.size(); // We may not be rendering all points being simulated.

    if (!pred)
        pred = trivial_predicate;

    for (int i = 0; i < N; i += stride)
    {
        for (int j = offset_into_inds[i]; j < offset_into_inds[i + 1]; ++j)
        {
            int const ds_idx = connectivity_inds[j];
            if (ds_idx % points_stride != 0)
                continue;

            float const* p1 = glm::value_ptr(g_attrs.points[i]);
            float const* p2 = glm::value_ptr(g_attrs.points[ds_idx / points_stride]);
            if (!pred(p1, p2))
                continue;

            g_attrs.lines.push_back(i);
            g_attrs.lines.push_back(ds_idx / points_stride);
        }
    }

    // Setup index buffer.
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_attrs.line_buff);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, g_attrs.lines.size() * sizeof(GLuint), g_attrs.lines.data(), GL_STATIC_DRAW);
}

void add_points(float const* points, int const N)
{
    g_attrs.points.insert(g_attrs.points.end(), (glm::vec3*)points, ((glm::vec3*)points) + N);
	glBindBuffer(GL_ARRAY_BUFFER, g_attrs.pos_buff);
	glBufferData(GL_ARRAY_BUFFER,
			g_attrs.points.size() * sizeof(glm::vec3),
			g_attrs.points.data(),
			GL_DYNAMIC_DRAW);
}

void add_lines(int const* lines, int const N)
{
    g_attrs.lines.insert(g_attrs.lines.end(), lines, lines + 2*N);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_attrs.line_buff);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, g_attrs.lines.size() * sizeof(GLuint), g_attrs.lines.data(), GL_STATIC_DRAW);
}

void add_triangles(int const* triangles, int const N)
{
    g_attrs.triangles.insert(g_attrs.triangles.end(), triangles, triangles + 3*N);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_attrs.tri_buff);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, g_attrs.triangles.size() * sizeof(GLuint), g_attrs.triangles.data(), GL_STATIC_DRAW);
}

bool render (float const* voltages, void const* slice, int const slice_width, int const slice_height, bool& pause)
{   
    if(glfwWindowShouldClose(g_attrs.window))
        return false; // Done!

    if (g_attrs.replay)
        voltages = g_attrs.replay_buffer.read_frame();

    glBindBuffer(GL_ARRAY_BUFFER, g_attrs.voltage_buff);
    if (!g_attrs.pause || g_attrs.replay) // Only update voltages during replay or when not paused. 
    {
        glBufferData(GL_ARRAY_BUFFER,
            g_attrs.voltages.size() * sizeof(GLfloat),
            voltages,
            GL_DYNAMIC_DRAW);
    }

    GLfloat const color[4]{ 0.0f, 0.0f, 0.0f, 1.0f };
    glClearBufferfv(GL_COLOR, 0.0f, color);
    glClear(GL_DEPTH_BUFFER_BIT);

    // Rotation of object.
	if (glfwGetMouseButton(g_attrs.window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
	{
		double xpos, ypos;
		glfwGetCursorPos(g_attrs.window, &xpos, &ypos);
		//    glfwSetCursorPos(g_attrs.window, g_attrs.width/2, g_attrs.height/2);
		glm::vec2 mouse{
			xpos - g_attrs.width / 2,
			g_attrs.height / 2 - ypos
		};
		g_attrs.theta = 0.001f * (mouse - g_attrs.mouse);

		auto rot_mat = glm::rotate(glm::mat4(1.0f), -g_attrs.theta.x, glm::vec3(0.0f, 1.0f, 0.0f));
		rot_mat = glm::rotate(rot_mat, -g_attrs.theta.y, glm::vec3(0.0f, 0.0f, 1.0f));
		g_attrs.cam_pos = glm::vec3(rot_mat * glm::vec4(g_attrs.cam_pos, 1.0f));
		g_attrs.cam_forward = glm::vec3(rot_mat * glm::vec4(g_attrs.cam_forward, 1.0f));
		g_attrs.cam_right = glm::vec3(rot_mat * glm::vec4(g_attrs.cam_right, 1.0f));

		g_attrs.view = glm::lookAt(g_attrs.cam_pos, g_attrs.look_pos, glm::vec3(0.0f, 1.0f, 0.0f));
		g_attrs.mouse = mouse;
	}
    else if (g_attrs.rotate)
    {
        auto rot_mat = glm::rotate(glm::mat4(1.0f), 0.003f, glm::vec3(0.0, 1.0, 0.0));
        g_attrs.cam_pos = glm::vec3(rot_mat * glm::vec4(g_attrs.cam_pos, 1.0f));
        g_attrs.view *= rot_mat;
    }

    // Shifting of object.
	if (glfwGetKey(g_attrs.window, GLFW_KEY_W) == GLFW_PRESS)
		g_attrs.model = glm::translate(g_attrs.model, 0.01f * glm::vec3(0, 0, 1));
	else if (glfwGetKey(g_attrs.window, GLFW_KEY_S) == GLFW_PRESS)
		g_attrs.model = glm::translate(g_attrs.model, -0.01f * glm::vec3(0, 0, 1));
	else if (glfwGetKey(g_attrs.window, GLFW_KEY_Q) == GLFW_PRESS)
		g_attrs.model = glm::translate(g_attrs.model, 0.01f * glm::vec3(0, 1, 0));
	else if (glfwGetKey(g_attrs.window, GLFW_KEY_E) == GLFW_PRESS)
		g_attrs.model = glm::translate(g_attrs.model, -0.01f * glm::vec3(0, 1, 0));
	else if (glfwGetKey(g_attrs.window, GLFW_KEY_A) == GLFW_PRESS)
		g_attrs.model = glm::translate(g_attrs.model, 0.01f * glm::vec3(1, 0, 0));
	else if (glfwGetKey(g_attrs.window, GLFW_KEY_D) == GLFW_PRESS)
		g_attrs.model = glm::translate(g_attrs.model, -0.01f * glm::vec3(1, 0, 0));

    // Bind voltage buffer and enable voltage attribute
    glEnableVertexAttribArray(g_attrs.pos_loc);
    glUseProgram(g_attrs.shaderProgram);
    glBindBuffer(GL_ARRAY_BUFFER, g_attrs.voltage_buff);
    glVertexAttribPointer(g_attrs.voltage_loc, 1, GL_FLOAT, GL_FALSE, sizeof(GLfloat), NULL);
    glEnableVertexAttribArray(g_attrs.voltage_loc);

    glUniformMatrix4fv(g_attrs.p_mat_loc, 1, GL_FALSE, glm::value_ptr(g_attrs.projection));
    glUniformMatrix4fv(g_attrs.m_mat_loc, 1, GL_FALSE, glm::value_ptr(g_attrs.model));
    glUniformMatrix4fv(g_attrs.v_mat_loc, 1, GL_FALSE, glm::value_ptr(g_attrs.view));
    glUniform3fv(g_attrs.campos_loc, 1, glm::value_ptr(g_attrs.cam_pos));

    glUniform1f(g_attrs.color_divisor_loc, g_attrs.color_divisor);
    glUniform1f(g_attrs.color_offset_loc, g_attrs.color_offset);
    glUniform1f(g_attrs.color_cutoff_loc, g_attrs.color_cutoff);

    if (g_attrs.draw_mode == 1 && !g_attrs.lines.empty())
    {
		// Draw the connectivity.
		glUniform1i(g_attrs.render_lines_loc, 1);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_attrs.line_buff);
		glDrawElements(GL_LINES, g_attrs.lines.size(), GL_UNSIGNED_INT, (void*)0);
		glUniform1i(g_attrs.render_lines_loc, 0);
    }

    if (g_attrs.draw_mode == 2 && !g_attrs.triangles.empty())
    {
		// Draw the triangles.
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_attrs.tri_buff);
		glDrawElements(GL_TRIANGLES, g_attrs.triangles.size(), GL_UNSIGNED_INT, (void*)0);
    }

    // Render neurons
    if(g_attrs.draw_mode < 2)
		glDrawArrays(GL_POINTS, 0, g_attrs.points.size());

    // Render texture
    if(slice)
    {
		g_attrs.slice_index %= slice_width;
        glDisableVertexAttribArray(g_attrs.voltage_loc);

        // Disable voltage attribute and enable tex_coords attribute
        glDisableVertexAttribArray(g_attrs.pos_loc);
        glDisableVertexAttribArray(g_attrs.voltage_loc);
        glUseProgram(g_attrs.tex_shaderProgram);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, g_attrs.texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, slice_width, slice_height, 0, GL_RED, GL_FLOAT, slice);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glUniform1i(g_attrs.sampler_loc, 0);

        float z_off{ (g_attrs.slice_index / (GLfloat)slice_width) * (g_attrs.world_max.z - g_attrs.world_min.z) };
        glm::mat4x4 tex_model_mat = g_attrs.model * glm::translate(glm::mat4x4(1.0f), glm::vec3(0.0f, 0.0f, z_off));
        glUniformMatrix4fv(g_attrs.p_mat_loc, 1, GL_FALSE, glm::value_ptr(g_attrs.projection));
        glUniformMatrix4fv(g_attrs.m_mat_loc, 1, GL_FALSE, glm::value_ptr(tex_model_mat));
        glUniformMatrix4fv(g_attrs.v_mat_loc, 1, GL_FALSE, glm::value_ptr(g_attrs.view));
        glDrawArrays(GL_TRIANGLES, 0, 6); 
    }

    glfwPollEvents();
    glfwSwapBuffers(g_attrs.window);
    pause = g_attrs.pause;
    return true;
}

void set_color_params(float divisor, float offset, float cutoff)
{
    g_attrs.color_divisor = divisor;
    g_attrs.color_offset = offset;
    g_attrs.color_cutoff = cutoff;

    glUniform1f(g_attrs.color_divisor_loc, g_attrs.color_divisor);
    glUniform1f(g_attrs.color_offset_loc, g_attrs.color_offset);
    glUniform1f(g_attrs.color_cutoff_loc, g_attrs.color_cutoff);
}

float get_color_divisor() { return g_attrs.color_divisor; }
float get_color_offset() { return g_attrs.color_offset; }
float get_color_cutoff() { return g_attrs.color_cutoff; }

void keyCallback(GLFWwindow* window, int key, int, int action, int)
{ 
#if 0 // Movement. Disabled for now. 
    if (key == GLFW_KEY_S && (action == GLFW_REPEAT || action == GLFW_PRESS))
        g_attrs.cam_pos -= 0.1f * g_attrs.dir;
    if (key == GLFW_KEY_W && (action == GLFW_REPEAT || action == GLFW_PRESS))
        g_attrs.cam_pos += 0.1f * g_attrs.dir;
    if (key == GLFW_KEY_A && (action == GLFW_REPEAT || action == GLFW_PRESS))
        g_attrs.cam_pos -= 0.1f * g_attrs.right;
    if (key == GLFW_KEY_D && (action == GLFW_REPEAT || action == GLFW_PRESS))
        g_attrs.cam_pos += 0.1f * g_attrs.right;
    if (key == GLFW_KEY_Q && (action == GLFW_REPEAT || action == GLFW_PRESS))
        g_attrs.cam_pos -= 0.1f * g_attrs.up;
    if (key == GLFW_KEY_E && (action == GLFW_REPEAT || action == GLFW_PRESS))
        g_attrs.cam_pos += 0.1f * g_attrs.up;
#endif 
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, 1);
    else if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    {
        g_attrs.pause = !g_attrs.pause; // Toggle pause
        if (!g_attrs.pause)
        {
            set_window_title(g_attrs.title.c_str());
            g_attrs.replay = false;
        }
        else
        {
            set_window_title((g_attrs.title + " ( PAUSED ) ").c_str());
        }
    }
    else if (key == GLFW_KEY_R && action == GLFW_PRESS)
        g_attrs.rotate = !g_attrs.rotate; // Toggle rotation

    else if (key == GLFW_KEY_LEFT_BRACKET && (action == GLFW_PRESS || action == GLFW_REPEAT))
    {
        g_attrs.color_divisor -= 0.1;
        std::cout << "Color divisor changed: " << g_attrs.color_divisor << std::endl;
        glUniform1f(g_attrs.color_divisor_loc, g_attrs.color_divisor);
    }
    else if (key == GLFW_KEY_RIGHT_BRACKET && (action == GLFW_PRESS || action == GLFW_REPEAT))
    {
        g_attrs.color_divisor += 0.1;
        std::cout << "Color divisor changed: " << g_attrs.color_divisor << std::endl;
        glUniform1f(g_attrs.color_divisor_loc, g_attrs.color_divisor);
    }
    else if (key == GLFW_KEY_SEMICOLON && (action == GLFW_PRESS || action == GLFW_REPEAT))
    {
        g_attrs.color_offset -= 0.1;
        std::cout << "color offset changed: " << g_attrs.color_offset << std::endl;
        glUniform1f(g_attrs.color_offset_loc, g_attrs.color_offset);
    }
    else if (key == GLFW_KEY_APOSTROPHE && (action == GLFW_PRESS || action == GLFW_REPEAT))
    {
        g_attrs.color_offset += 0.1;
        std::cout << "color offset changed: " << g_attrs.color_offset << std::endl;
        glUniform1f(g_attrs.color_offset_loc, g_attrs.color_offset);
    }
    else if (key == GLFW_KEY_COMMA && (action == GLFW_PRESS || action == GLFW_REPEAT))
    {
        g_attrs.color_cutoff -= 0.01;
        std::cout << "Color cutoff changed: " << g_attrs.color_cutoff << std::endl;
        glUniform1f(g_attrs.color_cutoff_loc, g_attrs.color_cutoff);
    }
    else if (key == GLFW_KEY_PERIOD && (action == GLFW_PRESS || action == GLFW_REPEAT))
    {
        g_attrs.color_cutoff += 0.01;
        std::cout << "Color cutoff changed: " << g_attrs.color_cutoff << std::endl;
        glUniform1f(g_attrs.color_cutoff_loc, g_attrs.color_cutoff);
    }

    else if (g_attrs.pause && key == GLFW_KEY_T && action == GLFW_PRESS) // Can only replay when paused! 
    {
        g_attrs.replay = !g_attrs.replay; // Toggle replay
        if (g_attrs.replay)
            set_window_title((g_attrs.title + " ( PAUSED ) ( REPLAY MODE ) -0").c_str());
        else
            set_window_title((g_attrs.title + " ( PAUSED ) ").c_str());
    }

    else if (g_attrs.replay)
    {
        if (key == GLFW_KEY_UP && (action == GLFW_REPEAT || action == GLFW_PRESS))
            g_attrs.replay_buffer.move_N_frames(1); // Fast forward replay one frame
        else if (key == GLFW_KEY_DOWN && (action == GLFW_REPEAT || action == GLFW_PRESS))
            g_attrs.replay_buffer.move_N_frames(-1); // Rewind replay one frame

        set_window_title((g_attrs.title + " ( PAUSED ) ( REPLAY MODE ) -" + std::to_string(g_attrs.replay_buffer.frame_rewind_idx)).c_str());
    }

    else if (key == GLFW_KEY_P && (action == GLFW_REPEAT || action == GLFW_PRESS))
        g_attrs.draw_mode = (g_attrs.draw_mode + 1) % 3;

    else if (key == GLFW_KEY_RIGHT && (action == GLFW_REPEAT || action == GLFW_PRESS))
        g_attrs.slice_index += 1;
    else if (key == GLFW_KEY_LEFT && (action == GLFW_REPEAT || action == GLFW_PRESS))
        g_attrs.slice_index -= 1;
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    double zoom_factor{ yoffset < 0.0 ? 1.05 : 0.95 };
    g_attrs.cam_pos *= zoom_factor;
    g_attrs.view = glm::lookAt(g_attrs.cam_pos, g_attrs.look_pos, glm::vec3(0.0f, 1.0f, 0.0f));
}

int setup(float*& voltages, float const* positions, float const* normals, size_t N, size_t render_N)
{
    {
        g_attrs.world_min = glm::vec3(-1.0);
        g_attrs.world_max = glm::vec3(1.0f);

        // setup render points 
        int render_sparsity{ (int)N / (int)render_N };
        g_attrs.points.resize(render_N);
        if(normals)
            g_attrs.normals.resize(render_N);
        for (int i = 0; i < render_N; ++i)
        {
            g_attrs.points[i] = ((glm::vec3*)positions)[i * render_sparsity];
            if(normals)
                g_attrs.normals[i] = ((glm::vec3*)normals)[i * render_sparsity];
        }

        cout << "Number of points: " << N << endl;
        cout << "Number of points for rendering: " << render_N << endl;

        g_attrs.voltages.resize(g_attrs.points.size(), 0.0f);
        startup(1920, 1080, "HazelViz");
     
        // Compile shaders for texture and voltages.
        initShaders("Shaders/vert.glsl", "Shaders/frag.glsl", g_attrs.shaderProgram);
        initShaders("Shaders/tex_vert.glsl", "Shaders/tex_frag.glsl", g_attrs.tex_shaderProgram);

        // Get locations of uniforms and attributes. 
        // Note that we assume p_mat_loc, v_mat_loc and m_mat_loc are same for tex shader and normal shader
        g_attrs.voltage_loc = glGetAttribLocation(g_attrs.shaderProgram, "voltage");
        g_attrs.pos_loc = glGetAttribLocation(g_attrs.shaderProgram, "position");
        g_attrs.norm_loc = glGetAttribLocation(g_attrs.shaderProgram, "normal");
        g_attrs.p_mat_loc = glGetUniformLocation(g_attrs.shaderProgram, "pMat");
        g_attrs.v_mat_loc = glGetUniformLocation(g_attrs.shaderProgram, "vMat");
        g_attrs.m_mat_loc = glGetUniformLocation(g_attrs.shaderProgram, "mMat");
        g_attrs.campos_loc = glGetUniformLocation(g_attrs.shaderProgram, "campos");
        g_attrs.render_lines_loc = glGetUniformLocation(g_attrs.shaderProgram, "render_lines");
        g_attrs.color_divisor_loc = glGetUniformLocation(g_attrs.shaderProgram, "color_divisor");
        g_attrs.color_cutoff_loc = glGetUniformLocation(g_attrs.shaderProgram, "color_cutoff");
        g_attrs.color_offset_loc = glGetUniformLocation(g_attrs.shaderProgram, "color_off");
        g_attrs.tex_coord_loc = glGetAttribLocation(g_attrs.tex_shaderProgram, "tex_coords");
        g_attrs.sampler_loc = glGetUniformLocation(g_attrs.tex_shaderProgram, "tex");

        // Load vertices into buffers. Also load uniform buffer.
        GLuint vao;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &g_attrs.pos_buff);
        glBindBuffer(GL_ARRAY_BUFFER, g_attrs.pos_buff);
        glBufferData(GL_ARRAY_BUFFER,
                g_attrs.points.size() * sizeof(glm::vec3),
                g_attrs.points.data(),
                GL_DYNAMIC_DRAW);

        glVertexAttribPointer(g_attrs.pos_loc, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(g_attrs.pos_loc);
        
        glGenBuffers(1, &g_attrs.voltage_buff);
        glBindBuffer(GL_ARRAY_BUFFER, g_attrs.voltage_buff);
        glBufferData(GL_ARRAY_BUFFER,
                g_attrs.voltages.size() * sizeof(GLfloat),
                g_attrs.voltages.data(),
                GL_DYNAMIC_DRAW);

        glGenBuffers(1, &g_attrs.norm_buff);
        glBindBuffer(GL_ARRAY_BUFFER, g_attrs.norm_buff);
        glBufferData(GL_ARRAY_BUFFER,
                g_attrs.normals.size() * sizeof(glm::vec3),
                g_attrs.normals.data(),
                GL_STATIC_DRAW);

        glVertexAttribPointer(g_attrs.norm_loc, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(g_attrs.norm_loc);

		glGenBuffers(1, &g_attrs.line_buff);

		glGenBuffers(1, &g_attrs.tri_buff);

        // Generate texture
        glGenTextures(1, &g_attrs.texture);

		glfwSetInputMode(g_attrs.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }
    // Init projection, view, model matrices
    g_attrs.projection = glm::perspective(glm::radians(45.0f),(GLfloat)g_attrs.width/g_attrs.height, 0.1f, 1000.0f); 
    g_attrs.cam_pos = glm::vec3(200, 0, 0);
    g_attrs.view = glm::lookAt(g_attrs.cam_pos, g_attrs.look_pos, glm::vec3(0, 1, 0)); 
    float scalar{ 30.0f };
    g_attrs.model = glm::scale(glm::mat4x4(1.0f), glm::vec3(scalar));

    glfwSetKeyCallback(g_attrs.window, keyCallback);
    glfwSetScrollCallback(g_attrs.window, scrollCallback);

    // Transparency stuff.
    glEnable (GL_BLEND); glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LESS);

    voltages = g_attrs.voltages.data();
    glPointSize(2);
    return 0;
}

void update_positions(float const* positions)
{
    glBindBuffer(GL_ARRAY_BUFFER, g_attrs.pos_buff);
    glBufferData(GL_ARRAY_BUFFER,
        g_attrs.points.size() * sizeof(glm::vec3),
        positions,
        GL_DYNAMIC_DRAW);
    glVertexAttribPointer(g_attrs.pos_loc, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(g_attrs.pos_loc);
}

void set_max_raw_video_size(int const size_in_gb)
{
    g_attrs.max_raw_video_size_gb = size_in_gb;
}

void write_frame(FILE* out, uint8_t* data)
{
    int w{ (int)g_attrs.width }, h{ (int)g_attrs.height };
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, data);
    // invert along y-axis
    for (int y = 0; y < h / 2; ++y)
        for (int x = 0; x < w; ++x)
            for (int chan = 0; chan < 3; ++chan)
                std::swap(data[3 * (y*w + x) + chan], data[3 * ((h - 1 - y)*w + x) + chan]);

    // Calculate size of file after it will be written to.
    g_attrs.raw_video_frame_idx += 1;
    size_t sz_in_bytes = w * h * 3 * sizeof(uint8_t) * g_attrs.raw_video_frame_idx;
    int const sz_in_gb = (int)(sz_in_bytes / 1e9);
    if (sz_in_gb >= g_attrs.max_raw_video_size_gb)
    {
        // Convert file to mov using batch file script.
        system(("convert.bat " + std::to_string(g_attrs.mov_video_idx)).c_str());
        g_attrs.mov_video_idx += 1;

        // Go back to start of raw data file. 
        fseek(out, 0, SEEK_SET);
        g_attrs.raw_video_frame_idx = 0;
    }

    fwrite(data, sizeof(uint8_t), w*h*3, out);
}

void setup_replay(int frame_cache_count)
{
    g_attrs.replay_buffer.init((int)g_attrs.points.size(), frame_cache_count);
}

void record_for_replay (float const* voltages)
{
    g_attrs.replay_buffer.write_frame(voltages);
}

void terminate_graphics ()
{
    glfwTerminate();
}

void set_window_title(const char* title) { glfwSetWindowTitle(g_attrs.window, title); }
int get_width() { return g_attrs.width; }
int get_height() { return g_attrs.height; }
int get_slice_index() { return g_attrs.slice_index; }

}; // Graphics namespacew
