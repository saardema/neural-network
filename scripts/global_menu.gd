extends MenuBar

@onready var settings_menu: PopupMenu = $SettingsMenu
var menu: MenuSystem = MenuSystem.new()
var show_fps: bool

func _ready():
	menu.add_toggle_item(settings_menu, "Show FPS", false, print, self , 'show_fps')
	menu.build()
