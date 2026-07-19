import { buildServer } from "./server.js";

const { app, services } = await buildServer();

try {
  const address = await app.listen({
    host: services.config.host,
    port: services.config.port,
  });
  app.log.info(`StructureFirst is available at ${address}`);
} catch (error) {
  app.log.error(error);
  process.exitCode = 1;
}
