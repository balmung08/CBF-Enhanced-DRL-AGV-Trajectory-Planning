#include <functional>

#include <gazebo/common/Events.hh>
#include <gazebo/gui/Actions.hh>
#include <gazebo/gui/GuiPlugin.hh>
#include <gazebo/rendering/rendering.hh>

namespace agv_4wis_gazebo
{

class HideCollisionsGuiPlugin : public gazebo::GUIPlugin
{
public:
  HideCollisionsGuiPlugin()
  {
    this->setObjectName("agv_hide_collisions_gui");
    this->resize(0, 0);
    pre_render_connection_ = gazebo::event::Events::ConnectPreRender(
      std::bind(&HideCollisionsGuiPlugin::hide_collisions, this));
  }

private:
  void hide_collisions()
  {
    if (gazebo::gui::g_showCollisionsAct) {
      gazebo::gui::g_showCollisionsAct->setChecked(false);
    }
    auto scene = gazebo::rendering::get_scene();
    if (scene) {
      scene->ShowCollisions(false);
      pre_render_connection_.reset();
    }
  }

  gazebo::event::ConnectionPtr pre_render_connection_;
};

GZ_REGISTER_GUI_PLUGIN(HideCollisionsGuiPlugin)

}  // namespace agv_4wis_gazebo
