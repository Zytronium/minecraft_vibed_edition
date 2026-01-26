use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Tool types that can be used to break blocks or as weapons
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ToolType {
    Sword,
    Axe,
    Pickaxe,
    Shovel,
    Hoe,
    Bow,
    Crossbow,
    None,
}

/// Tool tier determines effectiveness and durability
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ToolTier {
    Hand,
    Wood,
    Stone,
    Iron,
    Diamond,
    Netherite,
}

/// A single item that can drop from a block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Drop {
    #[serde(rename = "itemId")]
    pub item_id: String,
    pub min: u32,
    pub max: u32,
    /// Chance this drop occurs (0.0 to 1.0). If None, always drops.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chance: Option<f32>,
}

/// Collection of possible drops from a block
pub type DropTable = Vec<Drop>;

/// Texture configuration for a block's faces
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockTextures {
    /// Texture for all faces (shorthand)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub all: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bottom: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub north: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub south: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub east: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub west: Option<String>,
}

impl BlockTextures {
    /// Get the texture name for a specific face
    /// Face indices: 0=Left(West), 1=Right(East), 2=Bottom, 3=Top, 4=Back(North), 5=Front(South)
    pub fn get_texture(&self, face: usize) -> &str {
        if let Some(ref all) = self.all {
            return all;
        }

        match face {
            0 => self.west.as_deref().unwrap_or(""),
            1 => self.east.as_deref().unwrap_or(""),
            2 => self.bottom.as_deref().unwrap_or(""),
            3 => self.top.as_deref().unwrap_or(""),
            4 => self.north.as_deref().unwrap_or(""),
            5 => self.south.as_deref().unwrap_or(""),
            _ => "",
        }
    }
}

/// Block definition with all properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub id: String,
    pub name: String,

    /// Time to break this block
    pub hardness: f32,
    /// Resistance to explosions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resistance: Option<f32>,

    /// Tool type that mines this block faster
    #[serde(rename = "fasterTool", skip_serializing_if = "Option::is_none")]
    pub faster_tool: Option<ToolType>,
    /// Tool type required for block to drop anything
    #[serde(rename = "requiredTool", skip_serializing_if = "Option::is_none")]
    pub required_tool: Option<ToolType>,
    /// Minimum tool tier required for block to drop anything
    #[serde(rename = "minimumToolTier", skip_serializing_if = "Option::is_none")]
    pub minimum_tool_tier: Option<ToolTier>,

    pub drops: DropTable,
    pub textures: BlockTextures,

    pub transparent: bool,
    pub solid: bool,
    #[serde(rename = "lightLevel", skip_serializing_if = "Option::is_none")]
    pub light_level: Option<u8>, // 0-15

    /// Whether this block needs random ticks (plants, fire, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tickable: Option<bool>,
}

/// Base properties shared by all items
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseItem {
    pub id: String,
    pub name: String,
    pub texture: String,
    #[serde(rename = "maxStackSize")]
    pub max_stack_size: u32,
}

/// Tool item with combat and mining properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolItem {
    #[serde(flatten)]
    pub base: BaseItem,

    #[serde(rename = "toolType")]
    pub tool_type: ToolType,
    pub tier: ToolTier,

    pub damage: f32,
    pub durability: Option<u32>,
    #[serde(rename = "attackSpeed")]
    pub attack_speed: f32,
}

/// Block item that can be placed in the world
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockItem {
    #[serde(flatten)]
    pub base: BaseItem,

    #[serde(rename = "blockId")]
    pub block_id: String,
}

/// Food item that restores hunger/health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoodItem {
    #[serde(flatten)]
    pub base: BaseItem,

    pub hunger: i32,
    pub saturation: f32,
    pub health: i32,
}

/// Tagged union of all item types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum Item {
    #[serde(rename = "tool")]
    Tool(ToolItem),
    #[serde(rename = "block")]
    Block(BlockItem),
    #[serde(rename = "food")]
    Food(FoodItem),
}

impl Item {
    /// Get the base item properties regardless of item type
    pub fn base(&self) -> &BaseItem {
        match self {
            Item::Tool(t) => &t.base,
            Item::Block(b) => &b.base,
            Item::Food(f) => &f.base,
        }
    }
}

/// Recipe ingredient - either a specific item or a tag (group)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ingredient {
    #[serde(rename = "itemId", skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>, // defaults to 1
}

/// Result of a crafting recipe
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecipeResult {
    #[serde(rename = "itemId")]
    pub item_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>, // defaults to 1
}

/// Where a recipe can be crafted
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CraftingStation {
    CraftingTable,
    Stonecutter,
    Furnace,
    Smoker,
    Campfire,
    Anvil,
    None,
}

/// Condition for unlocking a recipe
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UnlockCondition {
    HasItem { has_item: String },
    HasTag { has_tag: String },
}

/// Shaped crafting recipe with a pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapedRecipe {
    /// Pattern grid (e.g. ["##", "# "] for 2x2)
    pub pattern: Vec<String>,
    /// Mapping from pattern symbols to ingredients
    pub key: HashMap<String, Option<Ingredient>>,
    pub result: RecipeResult,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub station: Option<CraftingStation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unlock: Option<Vec<UnlockCondition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
}

/// Shapeless crafting recipe (order doesn't matter)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapelessRecipe {
    pub ingredients: Vec<Ingredient>,
    pub result: RecipeResult,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub station: Option<CraftingStation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unlock: Option<Vec<UnlockCondition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
}

/// Smelting/cooking recipe
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmeltingRecipe {
    pub input: Ingredient,
    pub result: RecipeResult,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub experience: Option<f32>,
    #[serde(rename = "cookTime", skip_serializing_if = "Option::is_none")]
    pub cook_time: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub station: Option<CraftingStation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unlock: Option<Vec<UnlockCondition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
}

/// Tagged union of all recipe types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum CraftingRecipe {
    #[serde(rename = "shaped")]
    Shaped(ShapedRecipe),
    #[serde(rename = "shapeless")]
    Shapeless(ShapelessRecipe),
    #[serde(rename = "smelting")]
    Smelting(SmeltingRecipe),
}

/// Container for multiple blocks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlocksData {
    pub blocks: Vec<Block>,
}

/// Container for multiple items
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemsData {
    pub items: Vec<Item>,
}

/// Container for multiple recipes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecipesData {
    pub recipes: Vec<CraftingRecipe>,
}
