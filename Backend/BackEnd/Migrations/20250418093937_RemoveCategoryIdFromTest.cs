using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace BackEnd.Migrations
{
	/// <inheritdoc />
	public partial class RemoveCategoryIdFromTest : Migration
	{
		/// <inheritdoc />
		protected override void Up(MigrationBuilder migrationBuilder)
		{
			migrationBuilder.DropColumn(
				name: "CategoryId",
				table: "Test");
		}

		/// <inheritdoc />
		protected override void Down(MigrationBuilder migrationBuilder)
		{
			migrationBuilder.AddColumn<int>(
				name: "CategoryId",
				table: "Test",
				type: "integer",
				nullable: false,
				defaultValue: 0);
		}
	}
}