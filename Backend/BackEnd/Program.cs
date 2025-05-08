using Microsoft.EntityFrameworkCore;
using BackEnd;
using BackEnd.DBContext;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Configuration;

var builder = WebApplication.CreateBuilder(args);

// Добавление контекста базы данных
builder.Services.AddDbContext<ApplicationDbContext>(options =>
	options.UseNpgsql(builder.Configuration.GetConnectionString("DefaultConnection")));

// Добавление контроллеров
builder.Services.AddControllers();

// Настройка CORS (для работы с фронтендом, если он на другом порту)
builder.Services.AddCors(options =>
{
	options.AddPolicy("AllowAll", policy =>
	{
		policy.AllowAnyOrigin()
			  .AllowAnyMethod()
			  .AllowAnyHeader();
	});
});

var app = builder.Build();

// Настройка middleware
if (app.Environment.IsDevelopment())
{
	app.UseDeveloperExceptionPage();
}

// Включаем использование CORS
app.UseCors("AllowAll");

app.UseRouting();

// Роутинг и маппинг контроллеров
app.MapControllers();

app.Run();
