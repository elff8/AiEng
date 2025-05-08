using Microsoft.EntityFrameworkCore.Migrations;
using Npgsql.EntityFrameworkCore.PostgreSQL.Metadata;

#nullable disable

namespace BackEnd.Migrations
{
    /// <inheritdoc />
    public partial class ReplaceCategoryWithTestCategoryForTests : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<int>(
                name: "TestCategoryId",
                table: "Test",
                type: "integer",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.CreateTable(
                name: "TestCategories",
                columns: table => new
                {
                    Id = table.Column<int>(type: "integer", nullable: false)
                        .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityByDefaultColumn),
                    Name = table.Column<string>(type: "text", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_TestCategories", x => x.Id);
                });

            migrationBuilder.CreateIndex(
                name: "IX_Test_TestCategoryId",
                table: "Test",
                column: "TestCategoryId");

            migrationBuilder.AddForeignKey(
                name: "FK_Test_TestCategories_TestCategoryId",
                table: "Test",
                column: "TestCategoryId",
                principalTable: "TestCategories",
                principalColumn: "Id",
                onDelete: ReferentialAction.Cascade);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_Test_TestCategories_TestCategoryId",
                table: "Test");

            migrationBuilder.DropTable(
                name: "TestCategories");

            migrationBuilder.DropIndex(
                name: "IX_Test_TestCategoryId",
                table: "Test");

            migrationBuilder.DropColumn(
                name: "TestCategoryId",
                table: "Test");
        }
    }
}
